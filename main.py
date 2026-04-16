import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

# 导入我们的模型和处理类
from lstm_rl_model import DataProcessor, LSTMRLSystem, pmv_calculator, TimeSeriesDataset

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 设置绘图参数以支持中英文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 为中文字符
plt.rcParams['axes.unicode_minus'] = False  # 为负号显示

mae_criterion = nn.L1Loss()
def main():
    """主函数 - 数据导入和模型训练运行的完整流程"""

    print("========== LSTM-RL Energy Consumption Prediction System ==========")

    # 1. 加载数据
    print("\n1. Data Loading and Preprocessing")
    file_path = r'merged_data.csv'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' not found")
        return

    # 创建数据处理器
    data_processor = DataProcessor()

    # 加载原始数据
    df = data_processor.load_data(file_path)

    # 显示数据基本信息
    print("\nData Overview:")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Number of Data Points: {len(df)}")
    print(f"Original Features: {df.columns.tolist()}")

    # 2. 计算PMV值
    try:
        df = data_processor.calculate_and_add_pmv(df)
    except Exception as e:
        print(f"PMV calculation error: {e}")
        print("Skipping PMV calculation and proceeding with other processes")

    # 3. 添加时间特征
    df = data_processor.add_time_features(df)
    #df = data_processor.add_indoor_outdoor_temp_diff(df)

    # 4. 重采样到小时级别
    hourly_df = data_processor.resample_and_aggregate(df, freq='10min')
    #hourly_df = data_processor.resample_and_aggregate(df, freq='20min')
    print(f"Number of data points after resampling: {len(hourly_df)}")

    # 5. 检测并处理异常值
    print("\nDetecting and processing outliers:")
    energy_col = data_processor.col_mapping['energy']
    if energy_col in hourly_df.columns:
        hourly_df = data_processor.detect_outliers(hourly_df, energy_col, threshold=3)
    else:
        print(f"Warning: Energy column '{energy_col}' not found in dataset")

    # 6. 数据划分
    print("\n2. Data Splitting")

    total_days = (hourly_df.index.max() - hourly_df.index.min()).days + 1
    print(f"Total days: {total_days}")

    # 根据数据日期范围自动计算划分点
    lstm_train_end = hourly_df.index.min() + pd.Timedelta(days=int(total_days * 0.6))
    rl_train_end = hourly_df.index.min() + pd.Timedelta(days=int(total_days * 0.8))

    # 划分数据集
    lstm_train_data = hourly_df[hourly_df.index <= lstm_train_end]
    rl_train_data = hourly_df[(hourly_df.index > lstm_train_end) & (hourly_df.index <= rl_train_end)]
    test_data = hourly_df[hourly_df.index > rl_train_end]

    print(
        f"LSTM Training Set: {len(lstm_train_data)} samples ({lstm_train_data.index.min()} to {lstm_train_data.index.max()})")
    print(f"RL Training Set: {len(rl_train_data)} samples ({rl_train_data.index.min()} to {rl_train_data.index.max()})")
    print(f"Test Set: {len(test_data)} samples ({test_data.index.min()} to {test_data.index.max()})")

    # 7. 准备序列数据
    seq_length = 6  # 使用前24小时的数据预测下一小时
    energy_col = data_processor.col_mapping['energy']

    try:
        # 准备LSTM训练数据
        X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val, _, _ = data_processor.prepare_data(
            lstm_train_data,
            target_col=energy_col,
            seq_length=seq_length,
            test_split=0.0,  # 不需要测试集
            val_split=0.2  # 20%作为验证集
        )

        # 准备RL训练数据
        X_rl_train, y_rl_train, _, _, _, _ = data_processor.prepare_data(
            rl_train_data,
            target_col=energy_col,
            seq_length=seq_length,
            test_split=0.0,
            val_split=0.0  # 全部用于RL训练
        )

        # 准备测试数据
        X_test, y_test, _, _, _, _ = data_processor.prepare_data(
            test_data,
            target_col=energy_col,
            seq_length=seq_length,
            test_split=0.0,
            val_split=0.0  # 全部用于测试
        )
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return

    # 8. 创建并训练模型
    print("\n3. Model Training")

    # 定义输入形状
    input_shape = (seq_length, X_lstm_train.shape[2])

    # 创建LSTM-RL系统
    system = LSTMRLSystem(input_shape)

    # 训练LSTM基础模型
    print("\nTraining LSTM Base Model...")

    train_losses = []
    train_maes = []
    val_losses = []
    val_maes = []


    for epoch in range(20):
        system.lstm_model.train()
        train_loss = 0
        train_mae = 0
        train_dataset = TimeSeriesDataset(X_lstm_train, y_lstm_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.float().unsqueeze(1)

            system.lstm_model.optimizer.zero_grad()
            outputs = system.lstm_model(inputs)
            loss = system.lstm_model.criterion(outputs, labels)
            loss.backward()
            system.lstm_model.optimizer.step()
            train_loss += loss.item()
            mae = mae_criterion(outputs, labels)
            train_mae += mae.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        train_mae /= len(train_loader)
        train_maes.append(train_mae)

        system.lstm_model.eval()
        val_loss = 0
        val_mae = 0
        val_dataset = TimeSeriesDataset(X_lstm_val, y_lstm_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.float()
                labels = labels.float().unsqueeze(1)
                outputs = system.lstm_model(inputs)
                loss = system.lstm_model.criterion(outputs, labels)
                val_loss += loss.item()
                mae = mae_criterion(outputs, labels)
                val_mae += mae.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_mae /= len(val_loader)
        val_maes.append(val_mae)

        print(f'Epoch {epoch + 1}/100, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # # 绘制训练历史
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(train_maes, label='Training MAE')
    # plt.plot(val_maes, label='Validation MAE')
    # plt.title('Mean Absolute Error')
    # plt.xlabel('Epoch')
    # plt.ylabel('MAE')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('training_history.png')
    # plt.show()
    #
    # 训练RL代理
    print("\nTraining RL Agent...")
    system.train_rl_agent(
        X_rl_train, y_rl_train,
        episodes=100,
        batch_size=32
    )
    # 保存最佳LSTM参数状态
    print("保存LSTM最佳参数状态...")
    import pickle
    os.makedirs('models', exist_ok=True)

    # 保存当前LSTM参数作为最佳参数
    best_params = []
    for param in system.lstm_model.parameters():
        best_params.append(param.detach().clone())

    with open('models/best_lstm_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    # 保存数据处理器的scalers
    with open('models/data_scalers.pkl', 'wb') as f:
        pickle.dump({
            'feature_scalers': data_processor.feature_scalers,
            'energy_scaler': data_processor.energy_scaler,
            'col_mapping': data_processor.col_mapping
        }, f)

    print("最佳参数和数据标准化器已保存")
    system.lstm_model.save('models/rl_lstm_model.pth')

    # Enhanced professional plotting for IEEE journal
    import matplotlib.pyplot as plt
    import numpy as np

    # Set IEEE-style formatting
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.2

    # Create figure with IEEE column width (3.5 inches for single column)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Plot 1: Model Loss
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=1.5, marker='o', markersize=3, markevery=2)
    ax1.plot(val_losses, 'r--', label='Validation Loss', linewidth=1.5, marker='s', markersize=3, markevery=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('(a) Model Loss Convergence', fontsize=10, pad=15)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax1.set_xlim(0, len(train_losses) - 1)
    ax1.tick_params(axis='both', which='major', labelsize=9)

    # Add final loss values as annotations
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    ax1.annotate(f'{final_train_loss:.4f}',
                 xy=(len(train_losses) - 1, final_train_loss),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, ha='left')
    ax1.annotate(f'{final_val_loss:.4f}',
                 xy=(len(val_losses) - 1, final_val_loss),
                 xytext=(5, -5), textcoords='offset points',
                 fontsize=8, ha='left')

    # Plot 2: Mean Absolute Error
    ax2.plot(train_maes, 'b-', label='Training MAE', linewidth=1.5, marker='o', markersize=3, markevery=2)
    ax2.plot(val_maes, 'r--', label='Validation MAE', linewidth=1.5, marker='s', markersize=3, markevery=2)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('MAE', fontsize=10)
    ax2.set_title('(b) Mean Absolute Error', fontsize=10, pad=15)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax2.set_xlim(0, len(train_maes) - 1)
    ax2.tick_params(axis='both', which='major', labelsize=9)

    # Add final MAE values as annotations
    final_train_mae = train_maes[-1]
    final_val_mae = val_maes[-1]
    ax2.annotate(f'{final_train_mae:.4f}',
                 xy=(len(train_maes) - 1, final_train_mae),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, ha='left')
    ax2.annotate(f'{final_val_mae:.4f}',
                 xy=(len(val_maes) - 1, final_val_mae),
                 xytext=(5, -5), textcoords='offset points',
                 fontsize=8, ha='left')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save with high DPI for publication quality
    plt.savefig('training_history_ieee.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('training_history_ieee.eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')  # EPS format for LaTeX

    plt.show()

    # 9. 模型预测与评估
    print("\n4. Model Prediction and Evaluation")

    # 提取测试集日期
    #test_dates = test_data.index[seq_length:]
    # 提取测试集日期，确保与y_test长度一致
    test_dates = test_data.index[seq_length:seq_length + len(y_test)]
    print(f"test_dates length: {len(test_dates)}, y_test length: {len(y_test)}")

    # 使用仅LSTM模型预测
    print("\nPredicting with LSTM Model...")
    lstm_predictions = system.predict_with_lstm_only(X_test)

    # 使用RL增强的LSTM模型预测
    print("\nPredicting with RL-LSTM Model...")
    rl_lstm_predictions = system.predict_with_rl_lstm(X_test, y_test)

    # 评估函数定义
    def evaluate(y_true, predictions, label):
        """评估模型性能"""
        # 转换回原始尺度
        y_true_original = data_processor.inverse_transform_target(y_true).flatten()
        predictions_original = data_processor.inverse_transform_target(predictions).flatten()

        # 检查并处理NaN值
        nan_mask = np.isnan(predictions_original) | np.isnan(y_true_original)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} NaN values found in {label} evaluation, automatically removed")
            y_true_original = y_true_original[~nan_mask]
            predictions_original = predictions_original[~nan_mask]

        # 如果没有有效数据进行评估，则返回默认值
        if len(y_true_original) == 0:
            print(f"Error: No valid data for {label} evaluation")
            return {"rmse": float('nan'), "cvrmse": float('nan'), "mape": float('nan')}

        # 计算评价指标
        mse = mean_squared_error(y_true_original, predictions_original)
        rmse = np.sqrt(mse)

        # 计算CVRMSE
        mean_y = np.mean(y_true_original)
        cvrmse = rmse / mean_y

        # 计算MAPE
        mape = mean_absolute_percentage_error(y_true_original, predictions_original)

        print(f"\n{label} Model Evaluation Results:")
        print(f"RMSE: {rmse:.2f}")
        print(f"CVRMSE: {cvrmse:.2f}")
        print(f"MAPE: {mape:.2f}")

        return {
            "rmse": rmse,
            "cvrmse": cvrmse,
            "mape": mape
        }

    # 10. 评估模型性能
    print("\nModel Performance Evaluation:")

    # 评估LSTM模型
    lstm_metrics = evaluate(y_test, lstm_predictions, "LSTM")

    # 评估RL-LSTM模型
    rl_lstm_metrics = evaluate(y_test, rl_lstm_predictions, "RL-LSTM")

    # 11. 绘制预测结果对比图
    print("\nPlotting Prediction Results...")

    # 转换回原始尺度
    y_test_original = data_processor.inverse_transform_target(y_test).flatten()
    lstm_preds_original = data_processor.inverse_transform_target(lstm_predictions).flatten()
    rl_lstm_preds_original = data_processor.inverse_transform_target(rl_lstm_predictions).flatten()

    # plt.figure(figsize=(15, 8))
    # plt.plot(test_dates, y_test_original, label='Actual Values', color='blue', marker='o', markersize=4)
    # plt.plot(test_dates, lstm_preds_original, label='LSTM Predictions', color='green', marker='x', markersize=4)
    # plt.plot(test_dates, rl_lstm_preds_original, label='RL-LSTM Predictions', color='red', marker='^', markersize=4)
    #
    # plt.title('Energy Consumption Prediction Comparison', fontsize=16)
    # plt.xlabel('Date', fontsize=14)
    # plt.ylabel('Energy Consumption', fontsize=14)
    # plt.legend(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('energy_prediction_comparison.png', dpi=300)
    # plt.show()

    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    # Set IEEE-style formatting
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 1.0

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))

    # Main plot
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(test_dates, y_test_original, 'b-', label='Actual Values',
             linewidth=1.5, marker='o', markersize=2, markevery=10, alpha=0.8)
    ax1.plot(test_dates, lstm_preds_original, 'g--', label='LSTM Predictions',
             linewidth=1.5, marker='s', markersize=2, markevery=10, alpha=0.8)
    ax1.plot(test_dates, rl_lstm_preds_original, 'r:', label='RL-LSTM Predictions',
             linewidth=2, marker='^', markersize=2, markevery=10, alpha=0.8)

    ax1.set_ylabel('Energy Consumption (Wh)', fontsize=11)
    ax1.set_title('(a) Energy Consumption Prediction Comparison', fontsize=12, pad=15)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)

    # Format x-axis for dates
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.tick_params(axis='both', which='major', labelsize=9)

    # ===== CUSTOMIZE THESE DATES FOR YOUR DETAILED VIEW =====
    # Modify the dates below to match your data period of interest
    zoom_start_time = datetime.datetime(2024, 11, 19, 9, 0)  # Start: Nov 19, 8:00 AM
    zoom_end_time = datetime.datetime(2024, 11, 19, 18, 0)  # End: Nov 20, 8:00 PM
    # ========================================================

    # Find corresponding indices in your data
    try:
        zoom_start_idx = np.where(test_dates >= zoom_start_time)[0][0]
        zoom_end_idx = np.where(test_dates <= zoom_end_time)[0][-1]
        print(f"Selected time period: {zoom_start_time} to {zoom_end_time}")
        print(f"Data indices: {zoom_start_idx} to {zoom_end_idx}")
    except IndexError:
        print("Warning: Specified dates are outside data range.")
        print(f"Your data range: {test_dates[0]} to {test_dates[-1]}")
        # Use fallback selection (middle portion of data)
        zoom_start_idx = len(y_test_original) // 3
        zoom_end_idx = min(zoom_start_idx + 60, len(y_test_original) - 1)
        print(f"Using fallback: indices {zoom_start_idx} to {zoom_end_idx}")

    # Highlight zoom region on main plot
    zoom_start_date = test_dates[zoom_start_idx]
    zoom_end_date = test_dates[zoom_end_idx]
    y_min, y_max = ax1.get_ylim()
    zoom_rect = Rectangle((zoom_start_date, y_min), zoom_end_date - zoom_start_date,
                          y_max - y_min, linewidth=2, edgecolor='orange',
                          facecolor='yellow', alpha=0.2)
    ax1.add_patch(zoom_rect)

    # Add annotation for zoom region
    ax1.annotate('Detailed View', xy=(zoom_start_date, y_max * 0.9),
                 xytext=(10, -10), textcoords='offset points',
                 fontsize=9, ha='left', color='orange', weight='bold')

    # Zoom-in plot
    ax2 = plt.subplot(2, 1, 2)
    zoom_dates = test_dates[zoom_start_idx:zoom_end_idx]
    zoom_actual = y_test_original[zoom_start_idx:zoom_end_idx]
    zoom_lstm = lstm_preds_original[zoom_start_idx:zoom_end_idx]
    zoom_rl_lstm = rl_lstm_preds_original[zoom_start_idx:zoom_end_idx]

    # Enhanced visualization for better distinction
    ax2.plot(zoom_dates, zoom_actual, 'b-', label='Actual Values',
             linewidth=2.5, marker='o', markersize=5, alpha=0.9, zorder=3)
    ax2.plot(zoom_dates, lstm_preds_original[zoom_start_idx:zoom_end_idx], 'g--',
             label='LSTM Predictions', linewidth=2, marker='s', markersize=4,
             alpha=0.8, markerfacecolor='lightgreen', zorder=2)
    ax2.plot(zoom_dates, rl_lstm_preds_original[zoom_start_idx:zoom_end_idx], 'r:',
             label='RL-LSTM Predictions', linewidth=2.5, marker='^', markersize=4,
             alpha=0.9, markerfacecolor='lightcoral', zorder=1)

    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Energy Consumption (Wh)', fontsize=11)

    # Dynamic title based on selected period
    time_span = zoom_end_date - zoom_start_date
    ax2.set_title('(b) Detailed View: {} to {} ({:.1f} hours)'.format(
        zoom_start_date.strftime('%m-%d %H:%M'),
        zoom_end_date.strftime('%m-%d %H:%M'),
        time_span.total_seconds() / 3600), fontsize=12, pad=15)

    ax2.grid(True, alpha=0.4, linestyle=':')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)

    # Format x-axis for detailed view
    time_span_hours = time_span.total_seconds() / 3600
    if time_span_hours <= 8:
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    elif time_span_hours <= 24:
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    elif time_span_hours <= 72:
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=8))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    else:
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax2.tick_params(axis='both', which='major', labelsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Calculate and display performance metrics for zoom region
    if len(zoom_actual) > 0:
        mae_lstm_zoom = np.mean(np.abs(zoom_actual - zoom_lstm))
        mae_rl_lstm_zoom = np.mean(np.abs(zoom_actual - zoom_rl_lstm))
        mae_improvement = ((mae_lstm_zoom - mae_rl_lstm_zoom) / mae_lstm_zoom) * 100 if mae_lstm_zoom > 0 else 0

        rmse_lstm_zoom = np.sqrt(np.mean((zoom_actual - zoom_lstm) ** 2))
        rmse_rl_lstm_zoom = np.sqrt(np.mean((zoom_actual - zoom_rl_lstm) ** 2))
        rmse_improvement = ((rmse_lstm_zoom - rmse_rl_lstm_zoom) / rmse_lstm_zoom) * 100 if rmse_lstm_zoom > 0 else 0

        # Average model difference
        avg_model_diff = np.mean(np.abs(zoom_lstm - zoom_rl_lstm))
        max_model_diff = np.max(np.abs(zoom_lstm - zoom_rl_lstm))

        # # Add performance text box
        # textstr = f'Period: {len(zoom_actual)} data points\nModel Diff (Avg/Max): {avg_model_diff:.2f}/{max_model_diff:.2f} Wh\n\nLSTM MAE: {mae_lstm_zoom:.2f} Wh\nRL-LSTM MAE: {mae_rl_lstm_zoom:.2f} Wh\nMAE Improvement: {mae_improvement:.1f}%\n\nLSTM RMSE: {rmse_lstm_zoom:.2f} Wh\nRL-LSTM RMSE: {rmse_rl_lstm_zoom:.2f} Wh\nRMSE Improvement: {rmse_improvement:.1f}%'
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        # ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=8,
        #          verticalalignment='top', bbox=props)

    # Adjust layout
    plt.tight_layout()
    # Save with high quality
    plt.savefig('prediction_comparison_detailed.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('prediction_comparison_detailed.eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

    # 绘制预测误差对比
    plt.figure(figsize=(15, 6))

    # 计算预测误差
    lstm_errors = np.abs((y_test_original - lstm_preds_original) / y_test_original) * 100
    rl_lstm_errors = np.abs((y_test_original - rl_lstm_preds_original) / y_test_original) * 100

    # 处理可能的无限值或NaN值
    lstm_errors = np.nan_to_num(lstm_errors, nan=0.0, posinf=100.0, neginf=100.0)
    rl_lstm_errors = np.nan_to_num(rl_lstm_errors, nan=0.0, posinf=100.0, neginf=100.0)

    plt.bar(np.arange(len(test_dates)) - 0.2, lstm_errors, width=0.4, label='LSTM Prediction Error', color='green',
            alpha=0.7)
    plt.bar(np.arange(len(test_dates)) + 0.2, rl_lstm_errors, width=0.4, label='RL-LSTM Prediction Error', color='red',
            alpha=0.7)

    plt.title('Prediction Error Comparison (MAPE)', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Absolute Percentage Error (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_error_comparison.png', dpi=300)
    plt.show()

    # 12. 保存模型
    print("\nSaving Models...")
    os.makedirs('models', exist_ok=True)

    system.lstm_model.save('models/lstm_model.pth')
    system.rl_agent.save('models/rl_actor.pth', 'models/rl_critic.pth')

    print("\nModels have been saved to 'models' directory")
    print("\n========== Program Completed ==========")


if __name__ == "__main__":
    main()

