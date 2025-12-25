import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. 加载真实的学生成绩数据集
def load_student_data():
    """加载student_data_adjusted_rounded.csv文件并进行预处理"""
    try:
        # 读取CSV文件（自动适配常见编码）
        df = pd.read_csv('student_data_adjusted_rounded.csv', encoding='utf-8')
        print(f"✅ 成功加载数据，数据形状：{df.shape}")
        print(f"\n数据列名：{df.columns.tolist()}")
        print(f"\n数据前5行：\n{df.head()}")
        
        # 基本数据清洗
        # 删除空值行
        df = df.dropna()
        # 重置索引
        df = df.reset_index(drop=True)
        
        # 清理列名中的空格（关键修复：解决"作业完成 率"的空格问题）
        df.columns = df.columns.str.strip()
        
        return df
    except FileNotFoundError:
        print("❌ 错误：未找到student_data_adjusted_rounded.csv文件，请检查文件路径！")
        raise
    except Exception as e:
        print(f"❌ 数据加载出错：{str(e)}")
        raise

# 2. 数据预处理和特征工程
def preprocess_data(df):
    """数据预处理，生成模型所需的特征和目标变量"""
    # 定义核心特征列（扩展关键词，适配你的实际列名）
    feature_mapping = {
        '学习时长': ['每周学习时长（小时）', '学习时长', 'study_hours', 'hours_studied'],
        '出勤率': ['上课出勤率', '出勤率', 'attendance', 'attendance_rate'],
        '期中成绩': ['期中考试分数', '期中成绩', 'midterm_score', 'midterm'],
        '作业完成率': ['作业完成率', 'homework', 'homework_rate'],
        '专业': ['专业', 'major', 'department'],
        '性别': ['性别', 'gender', 'sex'],
        '期末成绩': ['期末考试分数', '期末成绩', 'final_score', 'final_grade', '期末分数']  # 关键修复：添加"期末考试分数"
    }
    
    # 自动匹配实际列名（统一转为小写，去除空格后匹配）
    cols_clean = [col.strip().lower() for col in df.columns]
    selected_cols = {}
    
    for key, possible_names in feature_mapping.items():
        for name in possible_names:
            # 将候选名也清理后匹配
            name_clean = name.strip().lower()
            if name_clean in cols_clean:
                # 找到匹配的原始列名
                original_idx = cols_clean.index(name_clean)
                original_col = df.columns[original_idx]
                selected_cols[key] = original_col
                break
    
    print(f"\n✅ 匹配到的列名：{selected_cols}")
    
    # 提取特征和目标变量
    # 数值特征
    numeric_features = []
    for key in ['学习时长', '出勤率', '期中成绩', '作业完成率']:
        if key in selected_cols:
            numeric_features.append(selected_cols[key])
    
    # 分类特征
    categorical_features = []
    for key in ['专业', '性别']:
        if key in selected_cols:
            categorical_features.append(selected_cols[key])
    
    # 目标变量（关键修复：严格检查）
    if '期末成绩' not in selected_cols:
        # 兜底：直接检查是否包含"期末"关键词的列
        final_cols = [col for col in df.columns if '期末' in col]
        if final_cols:
            selected_cols['期末成绩'] = final_cols[0]
            print(f"⚠️ 自动兜底匹配期末成绩列：{final_cols[0]}")
        else:
            raise ValueError("❌ 数据中未找到期末成绩相关列，请检查数据列名！")
    target_col = selected_cols['期末成绩']
    
    # 构建特征矩阵
    X_numeric = df[numeric_features].astype(float)
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[target_col].astype(float)
    
    print(f"\n✅ 特征矩阵形状：{X.shape}")
    print(f"✅ 目标变量形状：{y.shape}")
    print(f"\n特征列名：{X.columns.tolist()}")
    
    return X, y, X.columns.tolist()

# 3. 训练模型并保存
def train_and_save_model():
    """训练随机森林回归模型并保存"""
    # 加载数据
    df = load_student_data()
    
    # 预处理
    X, y, feature_names = preprocess_data(df)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    print("\n🚀 开始训练模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n✅ 模型训练完成！")
    print(f"📊 模型R²得分：{r2:.4f} (越接近1越好)")
    
    # 保存模型和特征名
    with open('score_prediction_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # 保存数据列名映射（用于预测时匹配）
    df_columns = df.columns.tolist()
    with open('data_columns.pkl', 'wb') as f:
        pickle.dump(df_columns, f)
    
    print("\n📁 已保存文件：")
    print("   - score_prediction_model.pkl (预测模型)")
    print("   - feature_names.pkl (特征列名)")
    print("   - data_columns.pkl (数据列名)")
    
    return model

# 执行训练
if __name__ == "__main__":
    train_and_save_model()
