import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 页面配置（必须放在最外层，且是第一个Streamlit命令）
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon="📚",
    layout="wide"
)

# 全局加载数据和模型
@st.cache_resource
def load_model_and_data():
    """加载训练好的模型和数据列名"""
    try:
        # 加载模型
        model = None
        feature_names = None
        model_path = 'score_prediction_model.pkl'
        feature_path = 'feature_names.pkl'
        
        if os.path.exists(model_path) and os.path.exists(feature_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(feature_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            st.warning("模型文件未找到，成绩预测功能将使用临时线性回归模型替代")
            # 临时训练一个简单模型（避免预测功能完全不可用）
            if os.path.exists('student_data_adjusted_rounded.csv'):
                df_temp = pd.read_csv('student_data_adjusted_rounded.csv', encoding='utf-8')
                df_temp.columns = df_temp.columns.str.strip()
                df_temp = df_temp.dropna()
                
                # 简单特征工程
                X = pd.get_dummies(df_temp[['每周学习时长（小时）', '上课出勤率', '期中考试分数', '作业完成率', '性别', '专业']], 
                                 columns=['性别', '专业'], drop_first=True)
                y = df_temp['期末考试分数']
                
                # 训练临时模型
                temp_model = LinearRegression()
                temp_model.fit(X, y)
                model = temp_model
                feature_names = X.columns.tolist()
        
        # 加载原始数据集（关键：清理列名空格）
        df = pd.read_csv('student_data_adjusted_rounded.csv', encoding='utf-8')
        df.columns = df.columns.str.strip()  # 清理列名空格
        df = df.dropna().reset_index(drop=True)
        
        # 数据预处理 - 添加百分比列
        df['上课出勤率_百分比'] = df['上课出勤率'] * 100
        df['作业完成率_百分比'] = df['作业完成率'] * 100
        
        return model, feature_names, df
    
    except FileNotFoundError as e:
        st.error(f"❌ 缺少必要文件：{str(e)}")
        st.info("请确保数据文件（student_data_adjusted_rounded.csv）存在于当前目录！")
        return None, None, None
    except Exception as e:
        st.error(f"❌ 加载模型/数据出错：{str(e)}")
        return None, None, None

# 加载模型和数据
model, feature_names, df = load_model_and_data()

# 1. 项目概述页面函数（优化图片加载容错）
def project_intro_page():
    """项目概述页面，展示系统介绍、目标、技术架构"""
    st.title("学生成绩分析与预测系统")
    
    # 项目概述 - 左侧文字，右侧可缩放示意图
    st.header("📖 项目概述")
    overview_col1, overview_col2 = st.columns([3,2])  #  比例分配空间
    
    with overview_col1:
        st.markdown("""
        本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。
        系统使用真实的学生成绩数据集（student_data_adjusted_rounded.csv）进行建模和分析，数据集包含5万条学生记录。
        
        ### 📈主要特点：
        - 📊 **数据可视化**：多维度展示学生学业数据
        - 🎯 **专业分析**：按专业/班级的详细统计分析
        - 🤖 **智能预测**：基于学习习惯预测成绩趋势
        - 💡 **学习建议**：根据预测结果提供个性化反馈
        """)
    
    with overview_col2:
        # 图片加载容错处理
        img_path = 'fenxi.PNG'
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path)
                st.image(image, caption='学生数据分析示意图', width="stretch")
                st.caption("💡 点击图片可放大查看")
            except Exception as e:
                st.warning(f"图片加载失败：{str(e)}")
                # 生成替代图表（Plotly版本）
                if df is not None:
                    sample_majors = df['专业'].value_counts().head(5).index
                    sample_data = df[df['专业'].isin(sample_majors)].groupby('专业')['期末考试分数'].mean().reset_index()
                    
                    fig = px.bar(
                        sample_data,
                        x='专业',
                        y='期末考试分数',
                        color='专业',
                        color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                        title='各专业平均期末成绩',
                        labels={'期末考试分数': '平均分数', '专业': '专业'},
                        height=300
                    )
                    fig.update_layout(
                        plot_bgcolor='white',
                        xaxis_tickangle=45,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("示意图文件（fenxi.PNG）未找到，显示替代图表")
            if df is not None:
                sample_majors = df['专业'].value_counts().head(5).index
                sample_data = df[df['专业'].isin(sample_majors)].groupby('专业')['期末考试分数'].mean().reset_index()
                
                fig = px.bar(
                    sample_data,
                    x='专业',
                    y='期末考试分数',
                    color='专业',
                    color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                    title='各专业平均期末成绩',
                    labels={'期末考试分数': '平均分数', '专业': '专业'},
                    height=300
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    xaxis_tickangle=45,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # 添加分隔横线
    st.markdown("---")
    
    # 项目目标
    st.header("🎯 项目目标")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯目标一：分析影响因素")
        st.markdown("""
        - 识别关键学习指标
        - 探索成绩相关因素
        - 提供数据支持决策
        """)
    
    with col2:
        st.subheader("🎯目标二：可视化展示")
        st.markdown("""
        - 专业对比分析
        - 性别差异研究
        - 学习模式识别
        """)
    
    with col3:
        st.subheader("🎯目标三：成绩预测")
        st.markdown("""
        - 机器学习建模
        - 个性化预测
        - 及时干预预警
        """)
        
    # 添加分隔横线
    st.markdown("---")
    
    # 技术架构
    st.header("🛠️ 技术架构")
    arch_col1, arch_col2, arch_col3, arch_col4 = st.columns(4)
    
    with arch_col1:
        st.info("**前端框架**")
        st.write("Streamlit")
    
    with arch_col2:
        st.info("**数据处理**")
        st.write("Pandas\nNumPy")
    
    with arch_col3:
        st.info("**可视化**")
        st.write("Plotly")
    
    with arch_col4:
        st.info("**机器学习**")
        st.write("Scikit-learn\n线性回归/随机森林")
        
    st.markdown("---")
    
    # 数据概览
    if df is not None:
        st.header("📊 数据概览")
        st.subheader("数据集基本信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总学生数", f"{len(df):,}")  # 千分位格式化
        with col2:
            st.metric("数据列数", len(df.columns))
        with col3:
            st.metric("专业数量", df['专业'].nunique())
        with col4:
            avg_score = df['期末考试分数'].mean()
            st.metric("平均期末成绩", f"{avg_score:.1f}")

# 2. 专业数据分析页面（全Plotly版本）
def major_analysis_page():
    """专业数据分析页面，展示各类统计图表"""
    if df is None:
        st.warning("⚠️ 暂无数据可供分析，请先加载数据集！")
        return
    
    st.title("📊 专业数据分析")
    
    # 筛选条件（固定在侧边栏，不影响主布局）
    st.sidebar.subheader("筛选条件")
    selected_majors = st.sidebar.multiselect(
        "选择专业", 
        df['专业'].unique(), 
        default=df['专业'].unique()
    )
    
    # 应用筛选
    filtered_df = df[df['专业'].isin(selected_majors)]
    
    # 数据量判断
    if len(filtered_df) == 0:
        st.warning("⚠️ 筛选后无数据，请调整筛选条件！")
        return
    
    # 按专业分组计算统计数据
    major_stats = filtered_df.groupby('专业').agg({
        '每周学习时长（小时）': 'mean',
        '期中考试分数': 'mean',
        '期末考试分数': 'mean',
        '上课出勤率_百分比': 'mean',
        '作业完成率_百分比': 'mean'
    }).round(2).reset_index()
    major_stats.columns = ['专业', '每周平均学时', '期中考试平均分', '期末考试平均分', '平均上课出勤率(%)', '平均作业完成率(%)']
    
    # 计算各专业性别比例
    gender_stats = pd.crosstab(filtered_df['专业'], filtered_df['性别'])
    gender_stats['总计'] = gender_stats.sum(axis=1)
    gender_stats['男生比例(%)'] = (gender_stats['男'] / gender_stats['总计'] * 100).round(2)
    gender_stats['女生比例(%)'] = (gender_stats['女'] / gender_stats['总计'] * 100).round(2)
    gender_stats = gender_stats.reset_index()
    
    # 核心可视化展示
    st.header("📈 学生学业表现可视化分析")
    
    # 1. 表格展示各专业统计数据
    st.subheader("1. 各专业核心统计数据")
    st.dataframe(major_stats.set_index('专业'), use_container_width=True)
    st.markdown("---")
    
    # 2. 性别比例柱状图 + 表格
    st.subheader("2. 各专业男女性别比例")
    chart1, table1 = st.columns([1, 1])
    
    with chart1:
        # 转换为长格式用于Plotly
        gender_long = pd.melt(
            gender_stats,
            id_vars=['专业'],
            value_vars=['男生比例(%)', '女生比例(%)'],
            var_name='性别',
            value_name='比例(%)'
        )
        
        fig1 = px.bar(
            gender_long,
            x='专业',
            y='比例(%)',
            color='性别',
            barmode='group',
            color_discrete_map={'男生比例(%)': '#3498db', '女生比例(%)': '#e74c3c'},
            title='各专业男女性别比例',
            labels={'比例(%)': '比例 (%)', '专业': '专业'},
            height=400
        )
        fig1.update_layout(
            plot_bgcolor='white',
            xaxis_tickangle=15,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        # 添加数值标签
        fig1.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    
    with table1:
        st.dataframe(
            gender_stats[['专业', '男', '女', '总计', '男生比例(%)', '女生比例(%)']].set_index('专业'),
            use_container_width=True,
            height=400
        )
    st.markdown("---")
    
    # 3. 期中期末分数对比折线图 + 表格
    st.subheader("3. 各专业期中/期末考试分数对比")
    chart2, table2 = st.columns([1, 1])
    
    with chart2:
        # 转换为长格式
        score_long = pd.melt(
            major_stats,
            id_vars=['专业'],
            value_vars=['期中考试平均分', '期末考试平均分'],
            var_name='考试类型',
            value_name='平均分'
        )
        
        fig2 = px.line(
            score_long,
            x='专业',
            y='平均分',
            color='考试类型',
            symbol='考试类型',
            color_discrete_map={'期中考试平均分': '#f39c12', '期末考试平均分': '#2ecc71'},
            title='各专业期中/期末考试分数对比',
            labels={'平均分': '平均分', '专业': '专业'},
            height=400,
            markers=True
        )
        fig2.update_layout(
            plot_bgcolor='white',
            xaxis_tickangle=15,
            yaxis_range=[0, 100],
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with table2:
        score_table = major_stats[['专业', '期中考试平均分', '期末考试平均分']].copy()
        score_table['分数提升'] = (score_table['期末考试平均分'] - score_table['期中考试平均分']).round(2)
        st.dataframe(
            score_table.set_index('专业'),
            use_container_width=True,
            height=400
        )
    st.markdown("---")
    
    # 4. 出勤率柱状图 + 表格
    st.subheader("4. 各专业平均上课出勤率")
    chart3, table3 = st.columns([1, 1])

    with chart3:
        # 改用离散深色（高饱和度），避免浅色问题
        fig3 = px.bar(
            major_stats,
            x='专业',
            y='平均上课出勤率(%)',
            color='专业',  # 按专业分配颜色（离散型）
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],  # 深色高饱和配色
            title='各专业平均上课出勤率',
            labels={'平均上课出勤率(%)': '出勤率 (%)', '专业': '专业'},
            height=400,
            text='平均上课出勤率(%)'
            )
        fig3.update_layout(
            plot_bgcolor='white',
            xaxis_tickangle=15,
            yaxis_range=[0, 100],
            showlegend=False  # 关闭图例（颜色已按专业区分，无需图例）
            )
        fig3.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            marker=dict(line=dict(color='white', width=1))  # 增加白色边框，增强视觉区分
            )
        st.plotly_chart(fig3, use_container_width=True)
    with table3:
        attendance_table = major_stats[['专业', '平均上课出勤率(%)', '平均作业完成率(%)']].copy()
        st.dataframe(
            attendance_table.set_index('专业'),
            use_container_width=True,
            height=400
        )
    st.markdown("---")
    
    # 5. 学习时长与成绩关系
    st.subheader("5. 学习时长 vs 期末成绩")
    fig4 = px.scatter(
        filtered_df,
        x='每周学习时长（小时）',
        y='期末考试分数',
        color='专业',
        trendline="ols",
        title="学习时长与成绩相关性",
        labels={'每周学习时长（小时）': '每周学习时长（小时）', '期末考试分数': '期末成绩'},
        opacity=0.7,
        height=600
    )
    fig4.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.05)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.05)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")
    
    # 6. 大数据管理专业专项分析
    st.subheader("6. 大数据管理专业专项分析")
    target_major = '大数据管理'
    if target_major in major_stats['专业'].values:
        bigdata_stats = major_stats[major_stats['专业'] == target_major].iloc[0]
        bigdata_df = filtered_df[filtered_df['专业'] == target_major].copy()
        
        # 步骤1：核心指标卡片
        st.subheader("核心指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="平均出勤率", value=f"{bigdata_stats['平均上课出勤率(%)']}%")
        with col2:
            st.metric(label="平均期末分数", value=f"{bigdata_stats['期末考试平均分']}分")
        with col3:
            # 计算通过率
            pass_count = len(bigdata_df[bigdata_df['期末考试分数'] >= 60])
            pass_rate = np.round((pass_count / len(bigdata_df) * 100), 1)
            st.metric(label="通过率", value=f"{pass_rate}%")
        with col4:
            st.metric(label="平均学习时长", value=f"{bigdata_stats['每周平均学时']}小时")
        
        # 步骤2：分布图表
        if len(bigdata_df) >= 3:
            st.subheader("数据分布")
            col_chart4, col_chart5 = st.columns([1, 1])
            
            # 左列：期末成绩分布直方图（Plotly）
            with col_chart4:
                st.subheader("期末成绩分布")
                bigdata_final_scores = pd.to_numeric(bigdata_df['期末考试分数'], errors='coerce').dropna()
                
                fig5 = px.histogram(
                    x=bigdata_final_scores,
                    nbins=min(10, len(bigdata_final_scores)-1) if len(bigdata_final_scores) > 1 else 5,
                    title='期末成绩分布',
                    labels={'x': '期末测试分数', 'y': '人数'},
                    color_discrete_sequence=['#3498db'],
                    height=350
                )
                fig5.update_layout(
                    plot_bgcolor='white',
                    bargap=0.1
                )
                st.plotly_chart(fig5, use_container_width=True)
            
            # 右列：学习时长分布箱线图（Plotly）
            with col_chart5:
                st.subheader("学习时长分布")
                bigdata_study_hours = pd.to_numeric(bigdata_df['每周学习时长（小时）'], errors='coerce').dropna()
                
                fig6 = px.box(
                    y=bigdata_study_hours,
                    title='学习时长分布',
                    labels={'y': '每周学习时长（小时）'},
                    color_discrete_sequence=['#3498db'],
                    height=350
                )
                fig6.update_layout(
                    plot_bgcolor='white',
                    xaxis_visible=False
                )
                st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info(f"ℹ️ 大数据管理专业仅{len(bigdata_df)}名学生，暂不展示分布图表（建议样本量≥3）")
    else:
        st.warning(f"⚠️ 未查询到【{target_major}】专业数据")
        st.info(f"当前数据包含的专业：{', '.join(df['专业'].unique())}")

# 3. 成绩预测页面（保持不变）
def score_prediction_page():
    """期末成绩预测页面，使用真实模型进行预测"""
    if model is None or df is None or feature_names is None:
        st.warning("⚠️ 模型或数据未加载成功，无法进行预测！")
        return
    
    st.title("🎯 期末成绩预测")
    st.markdown("请输入学生的学习信息，系统将使用机器学习模型预测其期末成绩并提供学习建议")
    
    # 表单输入
    with st.form('student_inputs'):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("学号", placeholder="例如：2023000001")
            gender = st.selectbox("性别", df['性别'].unique())
            major = st.selectbox("专业", df['专业'].unique())
        
        with col2:
            # 数值输入项（使用数据的真实范围，添加说明）
            study_hours = st.number_input(
                "每周学习时长（小时）", 
                min_value=float(df['每周学习时长（小时）'].min()), 
                max_value=float(df['每周学习时长（小时）'].max()), 
                step=0.5, 
                value=float(df['每周学习时长（小时）'].mean()),
                help=f"平均值：{df['每周学习时长（小时）'].mean():.1f}小时"
            )
            
            attendance = st.number_input(
                "上课出勤率", 
                min_value=float(df['上课出勤率'].min()), 
                max_value=float(df['上课出勤率'].max()), 
                step=0.01, 
                value=float(df['上课出勤率'].mean()),
                help=f"平均值：{df['上课出勤率'].mean():.2f}"
            )
            
            midterm_score = st.number_input(
                "期中考试分数", 
                min_value=float(df['期中考试分数'].min()), 
                max_value=float(df['期中考试分数'].max()), 
                step=1.0, 
                value=float(df['期中考试分数'].mean()),
                help=f"平均值：{df['期中考试分数'].mean():.1f}分"
            )
            
            homework_rate = st.number_input(
                "作业完成率", 
                min_value=float(df['作业完成率'].min()), 
                max_value=float(df['作业完成率'].max()), 
                step=0.01, 
                value=float(df['作业完成率'].mean()),
                help=f"平均值：{df['作业完成率'].mean():.2f}"
            )
        
        # 提交按钮
        submitted = st.form_submit_button("🔮 预测期末成绩", type="primary")
    
    # 预测逻辑
    if submitted:
        # 验证输入
        if not student_id:
            st.error("请输入学号！")
        else:
            try:
                # 构建输入数据
                input_data = {
                    '每周学习时长（小时）': study_hours,
                    '上课出勤率': attendance,
                    '期中考试分数': midterm_score,
                    '作业完成率': homework_rate,
                    '性别': gender,
                    '专业': major
                }
                
                # 转换为DataFrame
                input_df = pd.DataFrame([input_data])
                
                # 对分类特征进行独热编码（与训练时保持一致）
                input_df_encoded = pd.get_dummies(input_df, columns=['性别', '专业'], drop_first=True)
                
                # 确保输入特征与模型训练时一致
                for col in feature_names:
                    if col not in input_df_encoded.columns:
                        input_df_encoded[col] = 0
                
                # 只保留模型需要的特征列
                input_df_encoded = input_df_encoded[feature_names]
                
                # 使用模型预测
                prediction = model.predict(input_df_encoded)[0]
                # 限制在0-100分
                prediction = max(0, min(100, prediction))
                
                # 显示预测结果（使用卡片式布局）
                st.subheader("📊 预测结果")
                result_container = st.container(border=True)
                with result_container:
                    st.markdown(f"### 学号：{student_id}")
                    st.markdown(f"### 预测期末成绩：{prediction:.1f} 分")
                        
                    # 定义及格线（60分）
                    pass_score = 60
                    if prediction >= pass_score:
                        st.success("🎉 恭喜！你的期末成绩及格了！继续保持良好的学习习惯！")
                    else:
                        st.warning("💪 加油！你的期末成绩暂时不及格，但是只要努力就一定能进步！")
                
                # 显示对应图片
                success_img_path = "zhuhe.png"  # 及格图片路径
                encourage_img_path = "guli.jpeg"  # 不及格图片路径
                    
                if prediction >= pass_score:
                    # 显示恭喜图片
                    if os.path.exists(success_img_path):
                        try:
                            img = Image.open(success_img_path)
                            st.image(img, caption="恭喜你！继续加油！")
                        except Exception as e:
                            st.warning(f"恭喜图片加载失败: {str(e)}")
                    else:
                        st.warning(f"未找到恭喜图片，请确保{success_img_path}文件存在")
                else:
                    # 显示鼓励图片
                    if os.path.exists(encourage_img_path):
                        try:
                            img = Image.open(encourage_img_path)
                            st.image(img, caption="继续努力，一定能进步！")
                        except Exception as e:
                            st.warning(f"鼓励图片加载失败: {str(e)}")
                    else:
                        st.warning(f"未找到鼓励图片，请确保{encourage_img_path}文件存在")
                
                # 个性化学习建议
                st.subheader("💡 个性化学习建议")
                mean_study = df['每周学习时长（小时）'].mean()
                mean_attendance = df['上课出勤率'].mean()
                mean_homework = df['作业完成率'].mean()
                mean_midterm = df['期中考试分数'].mean()
                
                advice_container = st.container(border=True)
                with advice_container:
                    advice_list = []
                    if study_hours < mean_study:
                        advice_list.append(f"- ⏰ **增加学习时长**：当前{study_hours:.1f}小时，建议至少达到{mean_study:.1f}小时（平均水平）")
                    else:
                        advice_list.append(f"- ⏰ **学习时长**：当前{study_hours:.1f}小时，高于平均水平{mean_study:.1f}小时，继续保持！")
                    
                    if attendance < mean_attendance:
                        advice_list.append(f"- 🎒 **提高出勤率**：当前{attendance:.2f}，建议至少达到{mean_attendance:.2f}（平均水平）")
                    else:
                        advice_list.append(f"- 🎒 **出勤率**：当前{attendance:.2f}，高于平均水平{mean_attendance:.2f}，继续保持！")
                    
                    if homework_rate < mean_homework:
                        advice_list.append(f"- 📝 **完成作业**：当前{homework_rate:.2f}，建议至少达到{mean_homework:.2f}（平均水平）")
                    else:
                        advice_list.append(f"- 📝 **作业完成率**：当前{homework_rate:.2f}，高于平均水平{mean_homework:.2f}，继续保持！")
                    
                    if midterm_score < mean_midterm:
                        advice_list.append(f"- 📖 **查漏补缺**：当前期中{midterm_score:.1f}分，建议针对性复习薄弱环节（平均水平：{mean_midterm:.1f}分）")
                    else:
                        advice_list.append(f"- 📖 **期中考试**：当前{midterm_score:.1f}分，高于平均水平{mean_midterm:.1f}分，继续保持！")
                    
                    for advice in advice_list:
                        st.markdown(advice)
            
            except Exception as e:
                st.error(f"❌ 预测出错：{str(e)}")
                st.info("请检查输入数据是否合理，或刷新页面重试！")

# 侧边栏导航
st.sidebar.title("📑 导航菜单")
nav_option = st.sidebar.radio(
    "",  # 移除默认标题
    ["项目介绍", "专业数据分析", "成绩预测"],
    index=0
)

# 数据概览侧边栏
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 数据概览")
    st.sidebar.write(f"总学生数：{len(df):,}")
    st.sidebar.write(f"专业数量：{df['专业'].nunique()}")
    st.sidebar.write(f"男生数：{len(df[df['性别']=='男']):,}")
    st.sidebar.write(f"女生数：{len(df[df['性别']=='女']):,}")
    st.sidebar.write(f"平均成绩：{df['期末考试分数'].mean():.1f}分")

# 底部信息
st.sidebar.markdown("---")
st.sidebar.info("""
© 2025 学生成绩分析与预测系统  
📋 数据源：student_data_adjusted_rounded.csv  
💡 提示：筛选条件可在专业数据分析页面侧边栏调整
""")

# 导航逻辑
if nav_option == "项目介绍":
    project_intro_page()
elif nav_option == "专业数据分析":
    major_analysis_page()
elif nav_option == "成绩预测":
    score_prediction_page()
