import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from collections import deque
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. ROBUST DATA CLEANING
# ============================================

def load_and_clean_data(filepath):
    """Load and clean the dataset with robust handling"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Remove extreme outliers only (keep moderate volatility)
    for col in ['close', 'volume']:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df = df[(df[col] >= Q1) & (df[col] <= Q3)]
    
    # Convert date column
    if 'candidate' in df.columns:
        df['date'] = pd.to_datetime(df['candidate'])
        df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Cleaned dataset shape: {df.shape}")
    
    return df

# ============================================
# 2. ENHANCED FEATURE ENGINEERING WITH SAFETY
# ============================================

def safe_divide(a, b, default=0):
    """Safe division avoiding infinity"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = default
    return result

def prepare_features(df):
    """Prepare enhanced features with robust calculations"""
    print("\nPreparing enhanced features...")
    
    features = pd.DataFrame()
    
    # Basic features
    features['return_pct'] = df['return_pct']
    features['sentiment_score'] = df['sentiment_score']
    features['sentiment_label_enc'] = df['sentiment_label_enc']
    features['high_low_spread'] = df['high_low_spread']
    features['open_close_gap'] = df['open_close_gap']
    
    # Price momentum - clip extreme values
    features['price_change_3d'] = df['close'].pct_change(3).clip(-0.5, 0.5)
    features['price_change_5d'] = df['close'].pct_change(5).clip(-0.5, 0.5)
    features['price_change_10d'] = df['close'].pct_change(10).clip(-0.5, 0.5)
    
    # Volume indicators with safety
    vol_change = df['volume'].pct_change().clip(-2, 2)
    features['volume_change'] = vol_change
    
    vol_ma = df['volume'].rolling(10, min_periods=1).mean()
    features['volume_ma_ratio'] = safe_divide(df['volume'], vol_ma, 1.0)
    features['volume_ma_ratio'] = features['volume_ma_ratio'].clip(0, 5)
    
    # Volatility features
    features['volatility_5d'] = df['return_pct'].rolling(5, min_periods=1).std()
    features['volatility_10d'] = df['return_pct'].rolling(10, min_periods=1).std()
    features['volatility_20d'] = df['return_pct'].rolling(20, min_periods=1).std()
    
    # Technical indicators with safety
    if 'ret_mean_7d' in df.columns:
        features['ret_mean_7d'] = df['ret_mean_7d']
        features['ret_std_7d'] = df['ret_std_7d']
    
    # Momentum indicators
    features['momentum_5d'] = (df['close'] - df['close'].shift(5)).clip(-1000, 1000)
    features['momentum_10d'] = (df['close'] - df['close'].shift(10)).clip(-1000, 1000)
    features['momentum_20d'] = (df['close'] - df['close'].shift(20)).clip(-1000, 1000)
    
    # Sentiment momentum
    features['sentiment_change'] = df['sentiment_score'].diff().clip(-2, 2)
    features['sentiment_ma_3d'] = df['sentiment_score'].rolling(3, min_periods=1).mean()
    features['sentiment_ma_7d'] = df['sentiment_score'].rolling(7, min_periods=1).mean()
    
    # Lag features
    for i in [1, 2, 3, 5]:
        features[f'return_lag_{i}'] = df['return_pct'].shift(i)
        features[f'sentiment_lag_{i}'] = df['sentiment_score'].shift(i)
    
    # Price position relative to moving averages with safety
    close_ma_20 = df['close'].rolling(20, min_periods=1).mean()
    close_std_20 = df['close'].rolling(20, min_periods=1).std()
    features['price_vs_ma_20'] = safe_divide(df['close'] - close_ma_20, close_std_20, 0)
    features['price_vs_ma_20'] = features['price_vs_ma_20'].clip(-5, 5)
    
    # RSI-like indicator
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = safe_divide(gain, loss, 1.0)
    features['rsi_14'] = 100 - (100 / (1 + rs))
    features['rsi_14'] = features['rsi_14'].fillna(50)
    
    # Target: next day direction and magnitude
    features['target_direction'] = (df['next_day_return'] > 0).astype(int)
    features['target_magnitude'] = df['next_day_return'].abs().clip(0, 0.2)
    
    # Final cleaning - replace any remaining inf/nan
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Clip all features to reasonable ranges
    for col in features.columns:
        if col not in ['target_direction', 'target_magnitude']:
            q1 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = features[col].clip(q1, q99)
    
    print(f"Features shape: {features.shape}")
    print(f"Number of features: {len([c for c in features.columns if 'target' not in c])}")
    print(f"NaN count: {features.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}")
    
    return features

# ============================================
# 3. ADVANCED RL AGENT WITH BETTER ARCHITECTURE
# ============================================

class AdvancedTradingAgent:
    """Enhanced Deep Q-Network with better learning"""
    
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        
        # Improved hyperparameters
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Better network architecture
        self.weights = {}
        self.initialize_weights()
        
        # Performance tracking
        self.rewards_history = []
        self.accuracy_history = []
        self.portfolio_value = []
        self.actions_taken = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
    def initialize_weights(self):
        """Initialize deeper network with better initialization"""
        # Xavier initialization
        self.weights['W1'] = np.random.randn(self.state_size, 128) * np.sqrt(2.0 / self.state_size)
        self.weights['b1'] = np.zeros(128)
        self.weights['W2'] = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.weights['b2'] = np.zeros(64)
        self.weights['W3'] = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
        self.weights['b3'] = np.zeros(32)
        self.weights['W4'] = np.random.randn(32, self.action_size) * np.sqrt(2.0 / 32)
        self.weights['b4'] = np.zeros(self.action_size)
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, state):
        """Forward pass through deeper network"""
        h1 = self.relu(np.dot(state, self.weights['W1']) + self.weights['b1'])
        h2 = self.relu(np.dot(h1, self.weights['W2']) + self.weights['b2'])
        h3 = self.relu(np.dot(h2, self.weights['W3']) + self.weights['b3'])
        q_values = np.dot(h3, self.weights['W4']) + self.weights['b4']
        return q_values, h1, h2, h3
    
    def act(self, state, training=True):
        """Epsilon-greedy with better exploration"""
        if training and np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values, _, _, _ = self.forward(state)
        return np.argmax(q_values)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=128):
        """Enhanced training with better gradient updates"""
        if len(self.memory) < batch_size:
            return 0
        
        # Sample batch
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        total_loss = 0
        for state, action, reward, next_state, done in batch:
            # Current Q-values
            q_values, h1, h2, h3 = self.forward(state)
            target = q_values.copy()
            
            # Calculate target Q-value with double DQN concept
            if done:
                target[action] = reward
            else:
                next_q_values, _, _, _ = self.forward(next_state)
                target[action] = reward + self.gamma * np.max(next_q_values)
            
            # Calculate loss
            loss = np.mean((q_values - target) ** 2)
            total_loss += loss
            
            # Backpropagation with gradient clipping
            delta4 = (q_values - target)
            delta4 = np.clip(delta4, -1, 1)  # Gradient clipping
            
            grad_W4 = np.outer(h3, delta4)
            grad_b4 = delta4
            
            delta3 = np.dot(delta4, self.weights['W4'].T) * (h3 > 0)
            grad_W3 = np.outer(h2, delta3)
            grad_b3 = delta3
            
            delta2 = np.dot(delta3, self.weights['W3'].T) * (h2 > 0)
            grad_W2 = np.outer(h1, delta2)
            grad_b2 = delta2
            
            delta1 = np.dot(delta2, self.weights['W2'].T) * (h1 > 0)
            grad_W1 = np.outer(state, delta1)
            grad_b1 = delta1
            
            # Update weights with gradient clipping
            self.weights['W4'] -= self.learning_rate * np.clip(grad_W4, -1, 1)
            self.weights['b4'] -= self.learning_rate * np.clip(grad_b4, -1, 1)
            self.weights['W3'] -= self.learning_rate * np.clip(grad_W3, -1, 1)
            self.weights['b3'] -= self.learning_rate * np.clip(grad_b3, -1, 1)
            self.weights['W2'] -= self.learning_rate * np.clip(grad_W2, -1, 1)
            self.weights['b2'] -= self.learning_rate * np.clip(grad_b2, -1, 1)
            self.weights['W1'] -= self.learning_rate * np.clip(grad_W1, -1, 1)
            self.weights['b1'] -= self.learning_rate * np.clip(grad_b1, -1, 1)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size
    
    def calculate_reward(self, action, actual_return, sentiment, magnitude, volatility):
        """Sophisticated reward function for better learning"""
        reward = 0
        
        # Base reward for correct direction - scaled by magnitude
        if action == 1:  # BUY
            if actual_return > 0:
                reward = 5 + magnitude * 50  # Strong reward for profitable buys
            else:
                reward = -5 - magnitude * 30  # Penalty for losing buys
        elif action == 2:  # SELL
            if actual_return < 0:
                reward = 5 + magnitude * 50  # Strong reward for profitable sells
            else:
                reward = -5 - magnitude * 30  # Penalty for wrong sells
        else:  # HOLD
            if abs(actual_return) < 0.01:  # Correct hold during low movement
                reward = 2
            elif abs(actual_return) > 0.02:  # Missed opportunity
                reward = -3
            else:
                reward = 0
        
        # Sentiment alignment bonus
        if action == 1 and sentiment > 0.4:
            reward += 2
        elif action == 2 and sentiment < -0.3:
            reward += 2
        elif action == 1 and sentiment < -0.4:
            reward -= 1
        elif action == 2 and sentiment > 0.4:
            reward -= 1
        
        # Risk adjustment for volatility
        if action != 0 and volatility > 0.02:  # Risky environment
            if actual_return * (1 if action == 1 else -1) > 0:
                reward += 1  # Bonus for navigating volatility
        
        return reward

# ============================================
# 4. OPTIMIZED TRAINING
# ============================================

def train_agent(features, episodes=300, lookback=10):
    """Train with optimized parameters for 75%+ accuracy"""
    print("\n" + "="*50)
    print("TRAINING ADVANCED RL AGENT")
    print("="*50)
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    feature_cols = [col for col in features.columns if 'target' not in col]
    scaled_features = scaler.fit_transform(features[feature_cols])
    
    # Additional check
    assert not np.any(np.isnan(scaled_features)), "NaN in scaled features"
    assert not np.any(np.isinf(scaled_features)), "Inf in scaled features"
    
    # Initialize agent
    state_size = len(feature_cols)
    agent = AdvancedTradingAgent(state_size)
    
    # Training
    best_accuracy = 0
    patience = 0
    max_patience = 50
    
    for episode in range(episodes):
        total_reward = 0
        correct_predictions = 0
        total_predictions = 0
        portfolio_value = 10000
        
        for t in range(lookback, len(scaled_features) - 1):
            state = scaled_features[t]
            
            # Action
            action = agent.act(state, training=True)
            agent.actions_taken[['HOLD', 'BUY', 'SELL'][action]] += 1
            
            # Get actual return and features
            actual_return = features.iloc[t+1]['return_pct']
            magnitude = features.iloc[t+1]['target_magnitude']
            sentiment = features.iloc[t]['sentiment_score']
            volatility = features.iloc[t].get('volatility_10d', 0.01)
            
            # Calculate reward
            reward = agent.calculate_reward(action, actual_return, sentiment, magnitude, volatility)
            total_reward += reward
            
            # Update portfolio
            if action == 1:  # BUY
                portfolio_value *= (1 + actual_return)
            elif action == 2:  # SELL
                portfolio_value *= (1 - actual_return)
            
            # Next state
            next_state = scaled_features[t+1]
            done = (t == len(scaled_features) - 2)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Check accuracy
            actual_direction = features.iloc[t+1]['target_direction']
            
            if (action == 1 and actual_direction == 1) or \
               (action == 2 and actual_direction == 0) or \
               (action == 0 and abs(actual_return) < 0.01):
                correct_predictions += 1
            total_predictions += 1
            
            # Train multiple times per episode
            if len(agent.memory) >= 128:
                agent.replay(128)
        
        # Metrics
        accuracy = (correct_predictions / total_predictions) * 100
        agent.rewards_history.append(total_reward)
        agent.accuracy_history.append(accuracy)
        agent.portfolio_value.append(portfolio_value)
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience = 0
        else:
            patience += 1
        
        if (episode + 1) % 20 == 0:
            avg_acc_recent = np.mean(agent.accuracy_history[-20:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Acc: {accuracy:.2f}% | "
                  f"Avg(20): {avg_acc_recent:.2f}% | "
                  f"Portfolio: ${portfolio_value:.0f} | "
                  f"ε: {agent.epsilon:.3f}")
        
        # Stop if reached target
        if len(agent.accuracy_history) >= 50:
            recent_avg = np.mean(agent.accuracy_history[-50:])
            if recent_avg >= 75.0:
                print(f"\n🎯 Target reached! Average accuracy: {recent_avg:.2f}%")
                break
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    print(f"Final Accuracy: {agent.accuracy_history[-1]:.2f}%")
    print(f"Average Accuracy (last 50): {np.mean(agent.accuracy_history[-50:]):.2f}%")
    print(f"Final Portfolio: ${agent.portfolio_value[-1]:.2f}")
    print(f"Portfolio Return: {((agent.portfolio_value[-1] - 10000) / 10000) * 100:.2f}%")
    
    return agent, scaler

# ============================================
# 5. PREDICTIONS AND EXPLANATIONS
# ============================================

def make_predictions_with_explanations(agent, scaler, features, df, lookback=10):
    """Make predictions with detailed explanations"""
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    feature_cols = [col for col in features.columns if 'target' not in col]
    scaled_features = scaler.transform(features[feature_cols])
    
    predictions = []
    
    for t in range(max(lookback, len(scaled_features) - 30), len(scaled_features)):
        state = scaled_features[t]
        
        # Get Q-values
        q_values, _, _, _ = agent.forward(state)
        action = np.argmax(q_values)
        
        # Current data
        current = features.iloc[t]
        sentiment = current['sentiment_score']
        return_pct = current['return_pct']
        
        # Generate explanation
        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        action_str = action_map[action]
        
        # Sentiment analysis
        if sentiment > 0.5:
            sent_desc = "Strong positive sentiment"
            direction = "STRONG RISE expected"
        elif sentiment > 0.2:
            sent_desc = "Moderately positive sentiment"
            direction = "RISE expected"
        elif sentiment < -0.3:
            sent_desc = "Negative sentiment"
            direction = "FALL expected"
        else:
            sent_desc = "Neutral sentiment"
            direction = "SIDEWAYS movement"
        
        # Technical factors
        volatility = current.get('volatility_10d', 0)
        momentum = current.get('momentum_10d', 0)
        rsi = current.get('rsi_14', 50)
        
        vol_desc = "High volatility (risky)" if volatility > 0.015 else "Low volatility (stable)"
        mom_desc = "Strong positive momentum" if momentum > 100 else ("Positive momentum" if momentum > 0 else "Negative momentum")
        rsi_desc = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral RSI")
        
        explanation = f"{sent_desc}. {vol_desc}. {mom_desc}. {rsi_desc}. {direction}."
        
        # Confidence
        q_spread = np.max(q_values) - np.min(q_values)
        confidence = min(100, max(0, q_spread * 30))
        
        predictions.append({
            'index': t,
            'date': df.iloc[t]['candidate'] if 'candidate' in df.columns else t,
            'action': action_str,
            'confidence': f"{confidence:.1f}%",
            'close_price': df.iloc[t]['close'],
            'sentiment': f"{sentiment:.3f}",
            'return_pct': f"{return_pct*100:.2f}%",
            'rsi': f"{rsi:.1f}",
            'direction': direction,
            'explanation': explanation,
            'q_hold': f"{q_values[0]:.2f}",
            'q_buy': f"{q_values[1]:.2f}",
            'q_sell': f"{q_values[2]:.2f}"
        })
    
    return predictions

# ============================================
# 6. SAVE/LOAD MODEL
# ============================================

def save_model(agent, scaler, filepath='advanced_rl_agent.pkl'):
    """Save trained model"""
    print(f"\nSaving model to {filepath}...")
    model_data = {
        'weights': agent.weights,
        'epsilon': agent.epsilon,
        'state_size': agent.state_size,
        'rewards_history': agent.rewards_history,
        'accuracy_history': agent.accuracy_history,
        'portfolio_value': agent.portfolio_value,
        'scaler': scaler
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved!")

# ============================================
# 7. VISUALIZATION
# ============================================

def plot_results(agent):
    """Enhanced visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Accuracy
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(agent.accuracy_history, alpha=0.6, label='Accuracy', linewidth=1)
    ax1.plot(pd.Series(agent.accuracy_history).rolling(20).mean(), 
             linewidth=2, label='MA(20)', color='red')
    ax1.axhline(y=75, color='green', linestyle='--', linewidth=2, label='Target (75%)')
    ax1.axhline(y=33.33, color='gray', linestyle='--', label='Random Baseline')
    ax1.set_title('Accuracy Over Episodes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rewards
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(agent.rewards_history, color='green', alpha=0.7)
    ax2.plot(pd.Series(agent.rewards_history).rolling(20).mean(), 
             linewidth=2, color='darkgreen')
    ax2.set_title('Rewards', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.grid(True, alpha=0.3)
    
    # Portfolio Value
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(agent.portfolio_value, color='purple', linewidth=2)
    ax3.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Portfolio Value Over Episodes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.grid(True, alpha=0.3)
    final_value = agent.portfolio_value[-1]
    ax3.text(len(agent.portfolio_value)*0.7, final_value, 
             f'Final: ${final_value:.0f}', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat'))
    
    # Action Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    actions = list(agent.actions_taken.keys())
    counts = list(agent.actions_taken.values())
    colors = ['gray', 'green', 'red']
    ax4.bar(actions, counts, color=colors, alpha=0.7)
    ax4.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(counts):
        ax4.text(i, v, str(v), ha='center', va='bottom')
    
    # Accuracy Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(agent.accuracy_history, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax5.axvline(x=75, color='green', linestyle='--', linewidth=2)
    ax5.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Accuracy (%)')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)
    
    # Rewards Distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(agent.rewards_history, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax6.set_title('Rewards Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Reward')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    
    # Learning Summary
    ax7 = fig.add_subplot(gs[2, 2])
    window = 20
    ma_acc = pd.Series(agent.accuracy_history).rolling(window).mean()
    improvement = ma_acc.iloc[-1] - ma_acc.iloc[window] if len(ma_acc) > window else 0
    
    final_acc = agent.accuracy_history[-1]
    avg_acc_50 = np.mean(agent.accuracy_history[-50:]) if len(agent.accuracy_history) >= 50 else final_acc
    
    ax7.text(0.5, 0.75, f'Final Accuracy:\n{final_acc:.2f}%', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if final_acc >= 75 else 'lightyellow'))
    ax7.text(0.5, 0.5, f'Avg (Last 50):\n{avg_acc_50:.2f}%', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax7.text(0.5, 0.25, f'Improvement:\n{improvement:.2f}%', 
             ha='center', va='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat'))
    ax7.axis('off')
    ax7.set_title('Learning Summary', fontsize=12, fontweight='bold')
    
    plt.savefig('advanced_training_results.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as 'advanced_training_results.png'")
    plt.show()

def print_predictions(predictions):
    """Print formatted predictions"""
    print("\n" + "="*120)
    print("PREDICTIONS WITH EXPLANATIONS")
    print("="*120)
    
    for i, pred in enumerate(predictions[-15:], 1):
        print(f"\n{i}. {pred['date']}")
        print(f"   ACTION: {pred['action']} (Confidence: {pred['confidence']}) | Price: ${pred['close_price']:.2f} | Return: {pred['return_pct']}")
        print(f"   Sentiment: {pred['sentiment']} | RSI: {pred['rsi']} | Direction: {pred['direction']}")
        print(f"   Q-Values: HOLD={pred['q_hold']}, BUY={pred['q_buy']}, SELL={pred['q_sell']}")
        print(f"   💡 {pred['explanation']}")
        print("-" * 120)

# ============================================
# 8. MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED RL STOCK TRADING AGENT (TARGET: 75%+ ACCURACY)")
    print("="*70)
    
    DATA_FILE = 'preprocessed_nifty_sentiment.csv'
    EPISODES = 300
    LOOKBACK = 10
    
    try:
        # Load and prepare data
        df = load_and_clean_data(DATA_FILE)
        features = prepare_features(df)
        
        # Verify no inf/nan before training
        print("\nData quality check:")
        print(f"  - NaN values: {features.isna().sum().sum()}")
        print(f"  - Inf values: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Train agent
        agent, scaler = train_agent(features, episodes=EPISODES, lookback=LOOKBACK)
        
        # Generate predictions
        predictions = make_predictions_with_explanations(agent, scaler, features, df, lookback=LOOKBACK)
        
        # Save results
        save_model(agent, scaler)
        plot_results(agent)
        print_predictions(predictions)
        
        pd.DataFrame(predictions).to_csv('advanced_predictions.csv', index=False)
        print("\nPredictions saved to 'advanced_predictions.csv'")
        
        # Final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Episodes Trained: {len(agent.accuracy_history)}")
        print(f"Final Accuracy: {agent.accuracy_history[-1]:.2f}%")
        print(f"Best Accuracy: {max(agent.accuracy_history):.2f}%")
        
        if len(agent.accuracy_history) >= 50:
            avg_50 = np.mean(agent.accuracy_history[-50:])
            print(f"Avg Accuracy (last 50): {avg_50:.2f}%")
            if avg_50 >= 75.0:
                print("✅ TARGET ACHIEVED: 75%+ accuracy!")
            else:
                print(f"⚠️ Target not reached. Gap: {75.0 - avg_50:.2f}%")
        
        final_return = ((agent.portfolio_value[-1] - 10000) / 10000) * 100
        print(f"Final Portfolio: ${agent.portfolio_value[-1]:.2f}")
        print(f"Portfolio Return: {final_return:.2f}%")
        print(f"Total Actions: {sum(agent.actions_taken.values())}")
        print(f"  - BUY: {agent.actions_taken['BUY']} ({agent.actions_taken['BUY']/sum(agent.actions_taken.values())*100:.1f}%)")
        print(f"  - SELL: {agent.actions_taken['SELL']} ({agent.actions_taken['SELL']/sum(agent.actions_taken.values())*100:.1f}%)")
        print(f"  - HOLD: {agent.actions_taken['HOLD']} ({agent.actions_taken['HOLD']/sum(agent.actions_taken.values())*100:.1f}%)")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()