import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative, clo_dynamic
import random
import math
import warnings
warnings.filterwarnings('ignore')
# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


def pmv_calculator(time_series, Ta, Tg, humidity, wind_speed=0.1, M=1.1):
    """
    计算PMV（预测平均投票值）和Tm（平均辐射温度）

    参数:
    time_series: 时间戳序列
    Ta: 室内空气温度数组
    Tg: 黑球温度数组
    humidity: 相对湿度数组
    wind_speed: 风速，默认为0.1 m/s
    M: 新陈代谢率，默认为1.1 met

    返回:
    Tm_values: 平均辐射温度数组
    pmv_values: PMV值数组
    """
    Ta = np.asarray(Ta)
    Tg = np.asarray(Tg)
    humidity = np.asarray(humidity)

    def calculate_clothing_insulation(time_series):
        clo_values = []
        for timestamp in time_series:
            month = timestamp.month
            if 6 <= month <= 8:
                clo_values.append(0.5)  # Summer
            elif 9 <= month <= 11 or 3 <= month <= 5:
                clo_values.append(0.6)  # Intermediate seasons
            else:
                clo_values.append(0.8)  # Winter
        return np.array(clo_values)

    Tm_values = []
    for i in range(len(Tg)):
        try:
            Tm = ((float(Tg[i]) + 273) ** 4 + 2.5 * 10 ** 8 * abs(float(Tg[i]) - float(Ta[i])) ** (1 / 4) * (
                    float(Tg[i]) - float(Ta[i]))) ** (1 / 4) - 273
            Tm_values.append(float("{:.1f}".format(Tm)))
        except (ValueError, TypeError):
            print(f"计算平均辐射温度出错，索引 {i}：Ta={Ta[i]}, Tg={Tg[i]}")
            # 使用默认值
            Tm_values.append(Tg[i])

    Tm_values = np.array(Tm_values)
    clo = calculate_clothing_insulation(time_series)
    v = np.full_like(clo, wind_speed)
    vr = v_relative(v=v, met=M)
    clo_d = clo_dynamic(clo=clo, met=M)

    pmv_values = []
    for i in range(len(clo)):
        try:
            res = pmv(tdb=Ta[i], tr=Tm_values[i], vr=vr[i], rh=humidity[i], met=M, clo=clo_d[i])
            pmv_values.append(res)
        except (ValueError, TypeError):
            print(f"计算PMV出错，索引 {i}：Ta={Ta[i]}, Tm={Tm_values[i]}, RH={humidity[i]}")
            # 使用默认值
            pmv_values.append(0)

    pmv_values = np.array(pmv_values)
    return Tm_values, pmv_values


class DataProcessor:

    def __init__(self):
        self.energy_scaler = RobustScaler()  # 使用RobustScaler更好地处理异常值
        self.feature_scalers = {}  # 添加这一行来初始化feature_scalers字典

        # 定义列名映射
        self.col_mapping = {
            'time': 'account_datetime_jst',
            'indoor_temp': 'indoor_temperature',
            'indoor_humidity': 'indoor_humidity',
            'globe_temp': 'indoor_globe_temperature',
            'energy': 'total_electric[Wh]',
            'indoor_lux': 'indoor_lux',
            'indoor_co2': 'indoor_co2',
            'outdoor_temp': 'outdoor_temperature',
            'outdoor_humidity': 'outdoor_relativehumidity'
        }

    def load_data(self, file_path):
        """加载原始数据"""
        df = pd.read_csv(file_path)

        # 检查数据列
        print(f"数据集中的列名: {df.columns.tolist()}")

        # 使用指定的时间列
        time_column = self.col_mapping['time']

        # 检查时间列是否存在
        if time_column not in df.columns:
            raise KeyError(f"时间列 '{time_column}' 不存在于数据集中")

        # 将时间列转换为datetime格式
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.set_index(time_column)
        print(f"成功将 '{time_column}' 列设置为索引")

        # 检查缺失值并填充
        print(f"数据集中的缺失值数量: {df.isna().sum().sum()}")
        # 使用前向填充和后向填充组合处理缺失值
        # df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.ffill().bfill()

        return df

    # def add_indoor_outdoor_temp_diff(self, df):
    #     """添加室内外温差特征"""
    #     # 使用指定的列名
    #     indoor_temp_col = self.col_mapping['indoor_temp']
    #     outdoor_temp_col = self.col_mapping['outdoor_temp']
    #
    #     # 检查是否找到了所有必需的列
    #     missing_cols = []
    #     for col_name, col in zip(['室内温度', '室外温度'],
    #                              [indoor_temp_col, outdoor_temp_col]):
    #         if col not in df.columns:
    #             missing_cols.append(f"{col_name} ({col})")
    #
    #     if missing_cols:
    #         error_msg = f"计算室内外温差所需的列不存在: {', '.join(missing_cols)}"
    #         print(f"警告: {error_msg}")
    #         print("室内外温差计算将被跳过")
    #
    #         # 添加空的温差列
    #         df['indoor_outdoor_temp_diff'] = np.nan
    #         return df
    #
    #     # 计算室内外温差
    #     try:
    #         df['indoor_outdoor_temp_diff'] = df[indoor_temp_col] - df[outdoor_temp_col]
    #         print("室内外温差计算完成")
    #     except Exception as e:
    #         print(f"室内外温差计算出错: {e}")
    #         # 添加空的温差列
    #         df['indoor_outdoor_temp_diff'] = np.nan
    #
    #     return df

    def calculate_and_add_pmv(self, df):
        """计算并添加PMV和平均辐射温度到数据集"""
        print("计算PMV值...")

        # 使用指定的列名
        indoor_temp_col = self.col_mapping['indoor_temp']
        globe_temp_col = self.col_mapping['globe_temp']
        indoor_humidity_col = self.col_mapping['indoor_humidity']

        # 检查是否找到了所有必需的列
        missing_cols = []
        for col_name, col in zip(['室内温度', '黑球温度', '室内湿度'],
                                 [indoor_temp_col, globe_temp_col, indoor_humidity_col]):
            if col not in df.columns:
                missing_cols.append(f"{col_name} ({col})")

        if missing_cols:
            error_msg = f"计算PMV所需的列不存在: {', '.join(missing_cols)}"
            print(f"警告: {error_msg}")
            print("PMV计算将被跳过")

            # 添加空的PMV列
            df['mean_radiant_temperature'] = np.nan
            df['PMV'] = np.nan
            return df

        # 计算PMV和Tm
        try:
            Tm_values, pmv_values = pmv_calculator(
                df.index,
                df[indoor_temp_col].values,
                df[globe_temp_col].values,
                df[indoor_humidity_col].values
            )

            # 添加到数据集
            df['mean_radiant_temperature'] = Tm_values
            df['PMV'] = pmv_values
            print("PMV计算完成")
        except Exception as e:
            print(f"PMV计算出错: {e}")
            # 添加空的PMV列
            df['mean_radiant_temperature'] = np.nan
            df['PMV'] = np.nan

        return df

    # def add_time_features(self, df):
    #     """添加时间特征"""
    #     # 提取时间特征
    #     df['hour'] = df.index.hour
    #     df['dayofweek'] = df.index.dayofweek
    #     df['is_weekday'] = (df.index.dayofweek < 5).astype(int)
    #     df['day'] = df.index.day
    #     df['month'] = df.index.month
    #
    #     # 创建周期性特征
    #     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    #     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    #     df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    #     df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    #
    #     return df

    def add_time_features(self, df):
        """添加时间特征，包括日本节假日和分钟级特征"""
        import jpholiday  # 日本节假日库

        # 提取基本时间特征
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['is_weekday'] = (df.index.dayofweek < 5).astype(int)
        df['day'] = df.index.day
        df['month'] = df.index.month
        df['year'] = df.index.year

        # 添加分钟级周期特征
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

        # 添加小时级周期特征
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # 添加日本节假日特征
        df['is_holiday'] = df.index.map(lambda x: int(jpholiday.is_holiday(x.date())))

        # 添加月份的周期性特征
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def resample_and_aggregate(self, df, freq='H'):
        """重采样和聚合数据"""
        # 针对不同频率的数据聚合方法
        energy_col = self.col_mapping['energy']

        # 创建动态聚合字典
        agg_dict = {}

        # 对能耗列使用求和
        if energy_col in df.columns:
            agg_dict[energy_col] = 'sum'

        # 对所有其他列使用均值聚合
        for col in df.columns:
            if col != energy_col and col not in agg_dict:
                agg_dict[col] = 'mean'

        # 根据指定频率重采样
        # resampled_df = df.resample(freq).agg(agg_dict)
        resampled_df = df.resample(freq.replace('H', 'h')).agg(agg_dict)

        return resampled_df

    def detect_outliers(self, df, column, threshold=3):
        """检测并处理异常值"""
        # 检查列是否存在
        if column not in df.columns:
            print(f"警告：列 '{column}' 不存在于数据集中")
            return df

        # 使用IQR方法检测异常值（比Z分数更稳健）
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

        print(f"'{column}'列中检测到的异常值数量: {outliers.sum()}")

        # 将异常值替换为中位数
        if outliers.sum() > 0:
            df.loc[outliers, column] = df[column].median()
            print(f"异常值已被替换为中位数: {df[column].median()}")

        return df

    def create_sequences(self, data, target_col, seq_length=24):
        """创建时间序列数据"""
        X, y = [], []

        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, target_col])

        return np.array(X), np.array(y)

    def prepare_data(self, df, target_col=None, seq_length=24, test_split=0.2, val_split=0.1):
        """为模型训练和测试准备数据"""
        if target_col is None:
            target_col = self.col_mapping['energy']

        # 检查目标列是否存在
        if target_col not in df.columns:
            raise KeyError(f"目标列 '{target_col}' 不存在于数据集中")

        # 删除包含NaN值的行以确保数据干净
        df = df.dropna()
        print(f"删除NaN行后的形状: {df.shape}")

        # 分离特征和目标列
        features = df.drop(columns=[target_col])
        target = df[[target_col]]

        # 单独缩放每个特征
        scaled_features = np.zeros((len(features), features.shape[1]))
        for i, col in enumerate(features.columns):
            self.feature_scalers[col] = RobustScaler()
            scaled_features[:, i] = self.feature_scalers[col].fit_transform(features[[col]]).flatten()

        # 缩放目标
        scaled_target = self.energy_scaler.fit_transform(target)

        # 将特征和目标组合在一起
        combined_data = np.hstack((scaled_features, scaled_target))
        target_idx = scaled_features.shape[1]  # 目标列的索引

        # 创建时间序列数据
        X, y = self.create_sequences(combined_data, target_idx, seq_length)

        # 划分训练集、验证集和测试集
        train_size = int(len(X) * (1 - test_split - val_split))
        val_size = int(len(X) * val_split)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
        X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def inverse_transform_target(self, scaled_values):
        # Ensure predictions can't be negative
        if len(scaled_values.shape) == 1:
            scaled_values = scaled_values.reshape(-1, 1)

        # Apply inverse transform
        original_values = self.energy_scaler.inverse_transform(scaled_values)

        # Ensure non-negative values
        original_values = np.maximum(original_values, 0)

        return original_values


class LSTMModel(nn.Module):
    """LSTM基础模型类"""

    def __init__(self, input_shape, learning_rate=0.001):
        super(LSTMModel, self).__init__()
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_shape[1], 32, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(32 * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32 * 2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.bn(x[:, -1, :])
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练模型"""
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TimeSeriesDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        best_weights = None

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs = inputs.float()
                labels = labels.float().unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.float()
                    labels = labels.float().unsqueeze(1)
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.state_dict()

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        self.load_state_dict(best_weights)

    def predict(self, X):
        """进行预测"""
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self(X)
        predictions = predictions.numpy()
        predictions = np.maximum(predictions, 0)
        return predictions

    def save(self, file_path):
        """保存模型"""
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        """加载模型"""
        self.load_state_dict(torch.load(file_path))

    def get_gradients(self, X, y):
        """计算模型参数相对于预测误差的梯度"""
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.optimizer.zero_grad()
        outputs = self(X)
        loss = self.criterion(outputs, y)
        loss.backward()

        gradients = []
        for param in self.parameters():
            gradients.append(param.grad.clone())

        return gradients

    def update_weights(self, gradients, learning_factor):
        """使用梯度和学习因子更新模型参数"""
        for i, param in enumerate(self.parameters()):
            if gradients[i] is not None:
                if isinstance(learning_factor, torch.Tensor):
                    learning_factor = learning_factor.clone().detach()
                # If learning_factor is a numpy value or scalar
                else:
                    learning_factor = torch.tensor(learning_factor, dtype=torch.float)
                #learning_factor = torch.tensor(learning_factor)
                #learning_factor = torch.tensor(learning_factor, dtype=torch.float, device=self.device)
                param.data.sub_(learning_factor * gradients[i])


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.index = 0

    def add(self, state, action, reward, next_state):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """随机采样经验"""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_bound = action_bound

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x)) * self.action_bound
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(action_dim, 32)
        self.fc3 = nn.Linear(32 + 32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        state_out = self.relu(self.fc1(state))
        action_out = self.relu(self.fc2(action))
        concat = torch.cat([state_out, action_out], dim=1)
        x = self.relu(self.fc3(concat))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class DDPGAgent:
    """基于DDPG算法的强化学习代理"""

    def __init__(self, state_dim, action_dim, action_bound=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # 创建Actor和Critic网络
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)

        # 创建目标网络
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.target_critic = Critic(state_dim, action_dim)

        # 复制权重
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer()

        # 折扣因子
        self.gamma = 0.99

        # 目标网络软更新参数
        self.tau = 0.005

    def get_action(self, state, add_noise=True):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]

        # Severely limit action magnitude
        action = np.clip(action, -0.01, 0.01)  # Strict limits

        if add_noise:
            noise = np.random.normal(0, 0.005, size=self.action_dim)  # Reduced noise
            action = np.clip(action + noise, -0.01, 0.01)

        return action

    def add_experience(self, state, action, reward, next_state):
        """添加经验到回放缓冲区，确保数据格式一致"""
        # 确保state和next_state是一维数组
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten()

        # 确保action是一维数组
        if isinstance(action, (list, np.ndarray)):
            action = np.array(action, dtype=np.float32).flatten()
        else:
            action = np.array([action], dtype=np.float32)

        # 确保reward是标量
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[0])

        self.replay_buffer.add(state, action, reward, next_state)

    def train(self, batch_size=64):
        """从经验中学习"""
        if len(self.replay_buffer) < batch_size:
            return

        # 从回放缓冲区采样
        try:
            samples = self.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states = zip(*samples)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(next_states, dtype=torch.float32)

            # 训练Critic网络
            self.critic_optimizer.zero_grad()
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            target_q = rewards + self.gamma * target_q_values
            current_q = self.critic(states, actions)
            critic_loss = torch.mean((target_q - current_q) ** 2)
            critic_loss.backward()
            self.critic_optimizer.step()

            # 训练Actor网络
            self.actor_optimizer.zero_grad()
            actions = self.actor(states)
            critic_value = self.critic(states, actions)
            actor_loss = -torch.mean(critic_value)
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self._update_target_networks()
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            # 清空回放缓冲区以避免相同错误重复发生
            self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer.capacity)

    def _update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, actor_path, critic_path):
        """保存模型"""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path):
        """加载模型"""
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


class LSTMRLSystem:
    """LSTM-RL集成预测系统"""

    def __init__(self, input_shape):
        self.lstm_model = LSTMModel(input_shape)
        # self.data_processor = data_processor

        # RL代理的状态维度（这里使用预测误差）
        self.state_dim = 1

        # RL代理的动作维度（学习因子）
        self.action_dim = 1

        self.rl_agent = DDPGAgent(self.state_dim, self.action_dim)

        # 存储不同策略的预测结果
        self.lstm_predictions = None
        self.rl_lstm_predictions = None
        self.periodic_rl_predictions = None

        # 保存LSTM模型的当前权重
        self.original_weights = None

    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练LSTM基础模型"""
        print("训练LSTM基础模型...")
        self.lstm_model.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        # 保存LSTM原始权重
        self.original_weights = [param.detach().clone() for param in self.lstm_model.parameters()]

        return

    def train_rl_agent(self, X_rl_train, y_rl_train, episodes=100, batch_size=32):
        """训练RL代理"""
        print("训练RL代理...")

        for episode in range(episodes):
            total_reward = 0

            # # 重置LSTM模型权重到原始状态
            # for i, param in enumerate(self.lstm_model.parameters()):
            #     param.data.copy_(self.original_weights[i])

            # 初始预测
            initial_preds = self.lstm_model.predict(X_rl_train)
            initial_error = np.mean(np.abs((y_rl_train - initial_preds) / y_rl_train)) * 100
            state = np.array([initial_error])

            # 获取梯度
            gradients = self.lstm_model.get_gradients(X_rl_train, y_rl_train)

            # RL代理生成学习因子
            action = self.rl_agent.get_action(state)

            # 更新LSTM权重
            self.lstm_model.update_weights(gradients, action[0])

            # 再次预测
            updated_preds = self.lstm_model.predict(X_rl_train)
            updated_error = np.mean(np.abs((y_rl_train - updated_preds) / y_rl_train)) * 100
            next_state = np.array([updated_error])

            # # 在train_rl_agent方法中修改奖励计算
            # # 增加惩罚项，防止过度修改
            # original_params = np.concatenate([param.flatten().detach().numpy() for param in self.original_weights])
            # current_params = np.concatenate(
            #     [param.flatten().detach().numpy() for param in self.lstm_model.parameters()])
            # param_change = np.mean(np.abs(original_params - current_params))

            # 新的奖励函数，惩罚过大的参数变化
            reward = initial_error - updated_error

            # 计算奖励（误差减少则为正奖励）
            # reward = initial_error - updated_error
            total_reward += reward

            # 添加经验到回放缓冲区
            self.rl_agent.add_experience(state, action, reward, next_state)

            # 训练RL代理
            self.rl_agent.train(batch_size)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Error: {updated_error:.4f}%")

    def predict_with_lstm_only(self, X_test):
        """使用最佳LSTM参数进行预测"""
        import pickle
        import os

        # 尝试加载保存的最佳参数
        best_params_path = 'models/best_lstm_params.pkl'
        if hasattr(self, '_best_params_loaded') and self._best_params_loaded:
            # 已经加载过最佳参数，直接预测
            pass
        elif os.path.exists(best_params_path):
            try:
                with open(best_params_path, 'rb') as f:
                    best_params = pickle.load(f)

                # 加载最佳参数
                for i, param in enumerate(self.lstm_model.parameters()):
                    param.data.copy_(best_params[i])

                self._best_params_loaded = True
                print("已加载训练时的最佳LSTM参数")
            except Exception as e:
                print(f"加载最佳参数失败，使用当前参数: {e}")

        predictions = self.lstm_model.predict(X_test)
        return predictions

    def predict_with_rl_lstm(self, X_test, y_test):
        """使用RL增强的LSTM模型进行预测"""
        predictions = []

        # 重置LSTM模型权重到原始状态
        # for i, param in enumerate(self.lstm_model.parameters()):
        #     param.data.copy_(self.original_weights[i])

        for i in range(len(X_test)):
            # Reset weights every 6 hours instead of 24
            # if i % 6 == 0:
            #     for j, param in enumerate(self.lstm_model.parameters()):
            #         param.data.copy_(self.original_weights[j])

            # 使用当前模型进行预测
            X_current = X_test[i:i + 1]
            y_current = y_test[i:i + 1]

            initial_pred = self.lstm_model.predict(X_current)[0]
            predictions.append(initial_pred)

            if i < len(X_test) - 1:
                # 计算误差
                error = np.abs((y_current[0] - initial_pred) / y_current[0]) * 100
                state = np.array([error])

                # RL代理生成学习因子
                action = self.rl_agent.get_action(state, add_noise=False)

                # 获取梯度并更新权重
                gradients = self.lstm_model.get_gradients(X_current, y_current)
                self.lstm_model.update_weights(gradients, action[0])

        self.rl_lstm_predictions = np.array(predictions)
        return np.array(predictions)

    def predict_with_periodic_rl_update(self, X_test, y_test, update_interval=10):
        """使用定期更新的RL-LSTM模型进行预测 (简化版，不更新RL策略)"""
        # 初始化预测结果数组
        predictions = np.zeros_like(y_test)

        # 重置LSTM模型权重到原始状态
        for i, param in enumerate(self.lstm_model.parameters()):
            param.data.copy_(self.original_weights[i])

        for i in range(len(X_test)):
            try:
                # 当前输入和目标
                X_current = X_test[i:i + 1]
                y_current = y_test[i:i + 1]

                # 预测
                pred = self.lstm_model.predict(X_current)[0]

                # 检查NaN值
                if np.isnan(pred):
                    if i > 0:
                        print(f"警告: 样本 {i} 的预测为NaN，使用前一个预测")
                        pred = predictions[i - 1]
                    else:
                        print(f"警告: 样本 {i} 的预测为NaN，使用默认值0")
                        pred = 0.0

                predictions[i] = pred

                # 计算误差并更新模型
                if i < len(X_test) - 1:
                    # 防止除以零
                    if y_current[0] != 0:
                        error = np.abs((y_current[0] - pred) / y_current[0]) * 100
                    else:
                        error = np.abs(y_current[0] - pred) * 100

                    state = np.array([error], dtype=np.float32)

                    # RL代理生成学习因子
                    action = self.rl_agent.get_action(state, add_noise=False)

                    # 获取梯度并更新权重
                    gradients = self.lstm_model.get_gradients(X_current, y_current)
                    self.lstm_model.update_weights(gradients, action[0])
            except Exception as e:
                print(f"预测样本 {i} 时出错: {e}")
                # 使用前一个预测或默认值
                if i > 0:
                    predictions[i] = predictions[i - 1]
                else:
                    predictions[i] = 0.0

            # 定期打印进度
            if i > 0 and i % update_interval == 0:
                print(f"预测进度: {i}/{len(X_test)} 样本")

        self.periodic_rl_predictions = predictions
        return predictions



def periodic_rl_update_strategy(system, X_test, y_test, update_interval=10):
    """定期更新RL策略的预测函数

    参数:
    system: LSTMRLSystem实例
    X_test: 测试集特征
    y_test: 测试集目标
    update_interval: 更新RL策略的间隔(样本数)

    返回:
    predictions: 预测结果数组
    """
    return system.predict_with_periodic_rl_update(X_test, y_test, update_interval)
