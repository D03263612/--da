import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===================== 1. 全局配置项 =====================
# 请求头（模拟浏览器，真实爬取时使用）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://maoyan.com/"
}

# 档期映射（核心特征：区分春节/暑期/国庆/普通档）
SCHEDULE_MAP = {
    "春节档": ["01-21", "01-22", "01-23", "01-24", "01-25", "01-26", "01-27",
              "02-01", "02-02", "02-03", "02-04", "02-05", "02-06", "02-07"],  # 覆盖公历春节区间
    "暑期档": ["07-01", "08-31"],  # 7月1日-8月31日
    "国庆档": ["10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07"],
    "普通档": []
}

# 扩充基础数据列表（可生成上万条唯一电影数据）
BASE_MOVIE_PREFIX = ["流浪地球", "满江红", "唐探", "热辣滚烫", "封神", "飞驰人生", "长津湖", "水门桥",
                     "战狼", "你好李焕英", "人生大事", "万里归途", "悬崖之上", "狙击手", "无名",
                     "四海", "奇迹", "误杀", "怒火", "扫毒", "叶问", "功夫", "西游", "美人鱼"]
DIRECTOR_LIST = ["张艺谋", "吴京", "陈思诚", "冯小刚", "郭帆", "沈腾", "贾玲", "徐峥", "宁浩",
                 "林超贤", "陈凯歌", "王家卫", "周星驰", "徐克", "杜琪峰"]
FLOW_ACTORS = ["吴京", "沈腾", "易烊千玺", "张译", "贾玲", "刘德华", "黄渤", "沈腾", "周冬雨",
               "章子怡", "徐峥", "王宝强", "刘昊然", "吴京", "雷佳音"]
MOVIE_TYPES = ["动作", "喜剧", "科幻", "剧情", "悬疑", "战争", "爱情", "动画"]
YEAR_RANGE = (2020, 2025)  # 五年时间范围
TOTAL_MOVIE_NUM = 10000  # 目标数据量：1万条

# ===================== 2. 批量生成五年电影数据（替代真实爬取，避免反爬） =====================
def generate_batch_movie_names(total_num, year_range):
    """生成批量唯一电影名称（含年份标识）"""
    movie_names = []
    year_list = list(range(year_range[0], year_range[1]+1))
    for i in tqdm(range(total_num), desc="生成电影名称"):
        prefix = np.random.choice(BASE_MOVIE_PREFIX)
        suffix = str(np.random.randint(1, 10)) if np.random.random() > 0.3 else ""
        year = np.random.choice(year_list)
        movie_type = np.random.choice(MOVIE_TYPES)
        movie_name = f"{prefix}{suffix}（{movie_type}·{year}）"
        movie_names.append(movie_name)
    return movie_names

def generate_maoyan_movie_data(movie_names):
    """批量生成猫眼电影数据（票房、上映日期、排片率）"""
    maoyan_data = []
    for name in tqdm(movie_names, desc="生成猫眼数据"):
        # 提取电影年份
        year = re.findall(r"·(\d{4})", name)[0] if re.findall(r"·(\d{4})", name) else 2024
        # 随机生成上映日期
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29 if month in [2] else 31 if month in [1,3,5,7,8,10,12] else 30)
        release_date = f"{year}-{month:02d}-{day:02d}"
        # 票房分布：符合真实电影票房规律（大部分低票房，少数高票房）
        box_office = np.random.lognormal(2, 1.5) if np.random.random() > 0.1 else np.random.uniform(30, 60)
        # 排片率
        show_rate = np.random.uniform(5, 60)
        movie_info = {
            "电影名称": name,
            "上映年份": int(year),
            "上映日期": release_date,
            "总票房(亿)": round(box_office, 2),
            "排片率(%)": round(show_rate, 2),
            "日均排片场次(万场)": round(np.random.uniform(1, 20), 2)
        }
        maoyan_data.append(movie_info)
    return pd.DataFrame(maoyan_data)

def generate_douban_movie_data(movie_names):
    """批量生成豆瓣电影数据（评分、导演、主演、评论数）"""
    douban_data = []
    for name in tqdm(movie_names, desc="生成豆瓣数据"):
        # 豆瓣评分分布：集中在5-9分
        score = np.random.normal(7.0, 1.2)
        score = max(3.0, min(9.9, score))
        # 导演与主演
        director = np.random.choice(DIRECTOR_LIST)
        actor_num = np.random.randint(1, 5)
        actors = np.random.choice(FLOW_ACTORS, actor_num, replace=False).tolist()
        # 评论数
        comment_count = np.random.randint(1000, 1000000)
        movie_info = {
            "电影名称": name,
            "豆瓣评分": round(score, 1),
            "导演": director,
            "主演": ",".join(actors),
            "豆瓣评论数(条)": comment_count,
            "电影类型": np.random.choice(MOVIE_TYPES)
        }
        douban_data.append(movie_info)
    return pd.DataFrame(douban_data)

def generate_douyin_hot_data(movie_names):
    """批量生成抖音热度数据（话题播放量、点赞数、转发数）"""
    douyin_data = []
    for name in tqdm(movie_names, desc="生成抖音数据"):
        # 话题播放量：分布差异大
        play_count = np.random.lognormal(3, 2) if np.random.random() > 0.2 else np.random.uniform(50, 150)
        # 点赞数与转发数
        like_count = np.random.uniform(10, 2000)
        share_count = np.random.uniform(1, 500)
        movie_info = {
            "电影名称": name,
            "抖音话题播放量(亿)": round(play_count, 2),
            "抖音点赞数(万)": round(like_count, 2),
            "抖音转发数(万)": round(share_count, 2),
            "是否抖音热搜": 1 if play_count > 50 else 0
        }
        douyin_data.append(movie_info)
    return pd.DataFrame(douyin_data)

# ===================== 3. 大规模特征工程函数 =====================
def feature_engineering_large_scale(df):
    """大规模数据特征工程：档期分类、流量演员标记、时间特征、特征编码"""

    # 3.1 档期分类
    def get_schedule(date):
        """根据上映日期判断档期"""
        if pd.isna(date):
            return "普通档"
        month_day = date.split("-")[1:]
        month_day_str = "-".join(month_day)
        # 春节档
        if any(holiday in month_day_str for holiday in SCHEDULE_MAP["春节档"]):
            return "春节档"
        # 暑期档（7.1-8.31）
        elif SCHEDULE_MAP["暑期档"][0] <= month_day_str <= SCHEDULE_MAP["暑期档"][1]:
            return "暑期档"
        # 国庆档
        elif any(holiday in month_day_str for holiday in SCHEDULE_MAP["国庆档"]):
            return "国庆档"
        # 普通档
        else:
            return "普通档"

    df["档期类型"] = df["上映日期"].apply(get_schedule)

    # 3.2 流量演员标记（是否含顶流演员、顶流演员数量）
    def count_flow_actors(actor_str):
        """统计流量演员数量"""
        if pd.isna(actor_str):
            return 0
        return sum(1 for actor in FLOW_ACTORS if actor in actor_str)

    df["流量演员数量"] = df["主演"].apply(count_flow_actors)
    df["是否含流量演员"] = df["流量演员数量"].apply(lambda x: 1 if x >= 1 else 0)

    # 3.3 时间特征提取
    df["上映月份"] = df["上映日期"].apply(lambda x: int(x.split("-")[1]) if pd.notna(x) else np.nan)
    df["上映季度"] = df["上映月份"].apply(lambda x: (x-1)//3 + 1 if pd.notna(x) else np.nan)

    # 3.4 标签编码（分类特征转为数值）
    le_schedule = LabelEncoder()
    le_director = LabelEncoder()
    le_movie_type = LabelEncoder()

    df["档期类型编码"] = le_schedule.fit_transform(df["档期类型"].fillna("普通档"))
    df["导演编码"] = le_director.fit_transform(df["导演"].fillna("未知导演"))
    df["电影类型编码"] = le_movie_type.fit_transform(df["电影类型"].fillna("未知类型"))

    # 3.5 特征筛选（保留建模用核心特征，剔除冗余字段）
    feature_cols = [
        "豆瓣评分", "排片率(%)", "抖音话题播放量(亿)", "是否含流量演员", "档期类型编码",
        "导演编码", "电影类型编码", "上映季度", "流量演员数量", "抖音点赞数(万)", "是否抖音热搜"
    ]

    # 3.6 缺失值填充（确保模型可运行）
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df, feature_cols

# ===================== 4. 大规模数据模型构建与优化 =====================
def build_box_office_model_large_scale(df, feature_cols):
    """构建大规模数据XGBoost票房预测模型，分析特征重要性"""
    # 4.1 数据拆分（大规模数据下测试集占比10%即可）
    X = df[feature_cols]
    y = df["总票房(亿)"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )

    # 4.2 构建优化后的XGBoost模型（适配大规模数据）
    model = XGBRegressor(
        n_estimators=150,          # 增加树数量提升拟合能力
        max_depth=6,               # 适度增加树深度
        learning_rate=0.08,        # 减小学习率提升稳定性
        subsample=0.8,             # 样本抽样避免过拟合
        colsample_bytree=0.8,      # 特征抽样避免过拟合
        random_state=42,
        n_jobs=-1,                 # 启用多线程加速训练
        verbosity=0                # 静默训练
    )
    print("\n开始训练XGBoost模型（大规模数据，耐心等待...）")
    model.fit(X_train, y_train)

    # 4.3 模型评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    # 4.4 特征重要性（识别关键影响因素）
    feature_importance = pd.DataFrame({
        "特征名称": feature_cols,
        "重要性": model.feature_importances_
    }).sort_values("重要性", ascending=False)

    # 4.5 批量预测（全量数据）
    df["预测票房(亿)"] = model.predict(df[feature_cols])
    df["票房预测偏差(亿)"] = abs(df["总票房(亿)"] - df["预测票房(亿)"])

    # 4.6 模型评估指标汇总
    model_metrics = {
        "MAE（平均绝对误差）": round(mae, 4),
        "RMSE（均方根误差）": round(rmse, 4),
        "R2（拟合优度）": round(r2, 4)
    }

    return df, feature_importance, model_metrics

# ===================== 5. 大规模结果保存与汇总 =====================
def save_large_scale_results(df, feature_importance, model_metrics, save_path="./movie_box_office_10000_2020_2025.csv"):
    """保存1万条+五年电影数据的完整分析结果"""
    # 1. 保存特征重要性为独立CSV
    feature_importance_path = "./movie_feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")
    print(f"特征重要性已保存至：{feature_importance_path}")

    # 2. 保存模型评估指标为独立CSV
    metrics_df = pd.DataFrame([model_metrics])
    metrics_path = "./movie_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"模型评估指标已保存至：{metrics_path}")

    # 3. 保存全量数据（含原始特征+预测结果）
    # 筛选关键列，避免文件过大
    key_columns = [
        "电影名称", "上映年份", "上映日期", "档期类型", "总票房(亿)", "预测票房(亿)", "票房预测偏差(亿)",
        "豆瓣评分", "导演", "主演", "排片率(%)", "抖音话题播放量(亿)", "是否含流量演员",
        "模型MAE（平均绝对误差）", "模型RMSE（均方根误差）", "模型R2（拟合优度）"
    ]
    # 广播模型指标到全量数据
    for key, value in model_metrics.items():
        df[f"模型{key}"] = value

    # 保存全量数据
    df[key_columns].to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n1万条+五年电影全量分析结果已保存至：{save_path}")
    print(f"文件大小约：{round(df[key_columns].memory_usage(deep=True).sum()/1024/1024, 2)} MB")

    return df

# ===================== 6. 主执行流程（大规模数据生成与分析） =====================
if __name__ == "__main__":
    # 步骤1：生成1万条电影名称
    print("===== 开始生成1万条电影名称 =====")
    movie_list = generate_batch_movie_names(TOTAL_MOVIE_NUM, YEAR_RANGE)
    print(f"已生成 {len(movie_list)} 条电影名称")

    # 步骤2：批量生成多平台数据
    print("\n===== 开始生成多平台电影数据 =====")
    maoyan_df = generate_maoyan_movie_data(movie_list)
    douban_df = generate_douban_movie_data(movie_list)
    douyin_df = generate_douyin_hot_data(movie_list)

    # 步骤3：合并多源数据
    print("\n===== 开始合并多平台数据 =====")
    merged_df = maoyan_df.merge(douban_df, on="电影名称", how="inner")
    merged_df = merged_df.merge(douyin_df, on="电影名称", how="inner")
    print(f"合并后数据量：{len(merged_df)} 条")

    # 步骤4：大规模特征工程
    print("\n===== 开始大规模特征工程 =====")
    feature_df, feature_cols = feature_engineering_large_scale(merged_df)

    # 步骤5：构建XGBoost预测模型
    final_df, feature_importance, model_metrics = build_box_office_model_large_scale(feature_df, feature_cols)

    # 步骤6：输出关键结果
    print("\n===== 关键影响因素（特征重要性TOP10） =====")
    print(feature_importance.head(10))
    print("\n===== 模型评估指标 =====")
    for key, value in model_metrics.items():
        print(f"{key}：{value}")

    # 步骤7：保存所有结果
    print("\n===== 开始保存全量分析结果 =====")
    save_large_scale_results(final_df, feature_importance, model_metrics)

    print("\n===== 1万条+五年电影票房分析与预测任务全部完成 =====")
    print(f"数据时间范围：{YEAR_RANGE[0]} - {YEAR_RANGE[1]} 年")
    print(f"总数据量：{len(final_df)} 条")
    print(f"核心特征数：{len(feature_cols)} 个")
    print(f"模型拟合优度（R2）：{model_metrics['R2（拟合优度）']:.4f}")
