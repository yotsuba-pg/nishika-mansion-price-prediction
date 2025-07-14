# -*- coding: utf-8 -*-
"""
Created on Sun May 11 12:48:01 2025

@author: owara
"""

#　ライブラリのインポート
import pandas as pd#　表形式のデータを扱うためのライブラリ
import numpy as np#　数学的な処理に使うライブラリ
import datetime#　日付や時間を扱うためのライブラリ
import glob#　ファイルをまとめて読み込むためのライブラリ
import statsmodels.api as sm#スタッツモデルを使用
import matplotlib.pyplot as plt    
import japanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%%（データの読み込み）
# 指定したフォルダ内にある全てのcsvファイル読み込み。[rは「¥」をエラーなしで扱うための記法]
train_file_paths = glob.glob(r"C:\Users\owara\OneDrive\デスクトップ\Nishika\マンション\train\train.data\*.csv")
#　全てのcsvファイルを読み込み1つの表にまとめる。「utf-8」は文字化け防止のため
train_all_data = pd.concat([pd.read_csv(file, encoding="utf-8") for file in train_file_paths])
#　テストデータを読み込み。
test_df = pd.read_csv("test.csv", encoding="utf-8")

# 人口密度データ読み込み（都道府県ごとの人口密度データ）
density_df = pd.read_csv("都道府県別人口数（平成27年度）.csv", encoding="cp932")#　cp932は日本語Windows向けの文字コード。
#　必要な列（都道府県名と人口密度）だけを取り出し
density_df = density_df[['都道府県名', '人口密度（k㎡）']]
#　都道府県名の前後にある空白（スペース）を削除
density_df['都道府県名'] = density_df['都道府県名'].str.strip()

#%%

# =========================
# データ前処理関数
# =========================

#　preprecess_data とはひとまとまりの処理を定義
#　is_trainは訓練データかどうかを示すフラグ（テストデータを区別するため）
def preprocess_data(df, is_train=True):
    #　不要な列を指定して削除
    drop_columns = ["種類", "地域", "市区町村コード", "市区町村名",
                    "地区名", "最寄駅：名称", "土地の形状", "間口", "延床面積（㎡）",
                    "前面道路：方位", "前面道路：種類",
                    "前面道路：幅員（ｍ）", "都市計画", "取引時点", "取引の事情等"]
    
    # 間取り情報の前処理（まずはコピー直後に）
    # 「LDKあり」フラグ（LD, LDK, LDK+S, LDK+K など含むか）
    df["LDKあり"] = df["間取り"].astype(str).str.contains("LDK|ＬＤＫ|LD|ＬＤ").astype(int)

    # 「DKのみ」フラグ（DKを含み、LDKは含まない）
    df["DKのみ"] = df["間取り"].astype(str).str.contains("DK|ＤＫ").astype(int)
    df["DKのみ"] = df["DKのみ"] & (~df["LDKあり"].astype(bool))

    # 「Kのみ」フラグ（Kを含み、LDKもDKも含まない）
    df["Kのみ"] = df["間取り"].astype(str).str.contains("K|Ｋ").astype(int)
    df["Kのみ"] = df["Kのみ"] & (~df["LDKあり"].astype(bool)) & (~df["DKのみ"].astype(bool))

    # 「Rのみ」（ワンルーム：Rのみを含む）
    df["Rのみ"] = df["間取り"].astype(str).str.fullmatch(r".*R.*|.*Ｒ.*").astype(int)

    # 「Sあり」（納戸付き）
    df["Sあり"] = df["間取り"].astype(str).str.contains("S|Ｓ").astype(int)

    # 特殊な間取り（メゾネット・スタジオ・オープンフロアなど）
    df["特殊間取り"] = df["間取り"].astype(str).str.contains("メゾネット|スタジオ|オープンフロア").astype(int)

    # 「部屋数」数値抽出（例: 3LDK → 3）
    df["部屋数"] = df["間取り"].astype(str).str.extract(r"(\d+)").astype(float)

    # 元の「間取り」列は削除
    df = df.drop(columns=["間取り"], errors="ignore")
    
    # 建物構造のダミー変数化（SRC、RCを対象）
    df["構造_SRC"] = df["建物の構造"].astype(str).str.contains("ＳＲＣ").astype(int)
    df["構造_RC"] = df["建物の構造"].astype(str).str.contains("ＲＣ").astype(int)

    # 元の列は削除（他構造は無視）
    df = df.drop(columns=["建物の構造"], errors="ignore")
         
    #　都道府県名の整形と人口密度との結合
    #　都道府県名の前後のスペースを削除
    df['都道府県名'] = df['都道府県名'].str.strip()
    #　merge（読み込んだデータをdfに結合）、onは"都道府県名”は結合する共通のキー、how="left"は元の結合する共通のキー
    df = pd.merge(df, density_df, on='都道府県名', how='left')
    
    #　訓練データの時だけ行う処理
    if is_train:
        #　欠損値(NaN）がある行を削除　　
        df = df.dropna(subset=["面積（㎡）", "建築年", "最寄駅：距離（分）", "取引価格（総額）_log"])
        #　価格が0以下のデータは削除
        df = df[df["取引価格（総額）_log"] > 0]
    
    #%%
    # 面積処理
    #　「~㎡以上」という文字を削除、数値型（float）に変換
    df["面積（㎡）"] = df["面積（㎡）"].astype(str).str.replace("㎡以上", "", regex=True).astype(float)
    #　面積を対数変換（log）#log変換は極端に大きい/小さい値の影響を抑えるために実施
    df["面積（㎡）"] = np.log(df["面積（㎡）"])
    
    #%%
    # 和暦 → 西暦変換するための関数
    def wareki_to_seireki(wareki):
        #　年を削除
        if isinstance(wareki, str):
            wareki = wareki.replace("年", "")
            #　それぞれ和暦開始前の年を入力して、開始した年（元年）を+1と表現
            if "昭和" in wareki:
                return 1925 + int(wareki.replace("昭和", ""))
            elif "平成" in wareki:
                return 1988 + int(wareki.replace("平成", ""))
            elif "令和" in wareki:
                return 2018 + int(wareki.replace("令和", ""))
            #　戦前と描かれている場合は1945年とみなし、その他はNaN（欠損値）
            elif wareki == "戦前":
                return 1945
        return np.nan
    
    #　建築年・建築年の計算
    #　66行で作成した関数を用いて、和暦から西暦に変換した列を作成
    df["建築年_西暦"] = df["建築年"].apply(wareki_to_seireki).astype(float)
    #　現在の年から引いて築年数を求める
    df["築年数"] = datetime.datetime.now().year - df["建築年_西暦"]
    
    #%%
    #　欠損値の補完
    # 最寄駅名ごとの建築年_西暦の平均値で埋める
    mean_year_by_station = df.groupby("最寄駅：名称")["建築年_西暦"].mean()
    #　データ全体の「建築年_西暦」の中央値で計算（これは補完できない場合の予備的な値）
    overall_median_year = df["建築年_西暦"].median()
    
    #　補完の関数を定義
    #　建築年_西暦が欠損している（NaN）の場合は、その最寄駅の平均年を使用、なければ全体の中央値で補う
    def fill_missing_year(row):
        if pd.isnull(row["建築年_西暦"]):
            station_name = row["最寄駅：名称"]
            return mean_year_by_station.get(station_name, overall_median_year)
        #　欠損でなければそのままの値を使う
        return row["建築年_西暦"]
    #　99行で作成した補完関数を全業に対して適用
    df["建築年_西暦"] = df.apply(fill_missing_year, axis=1)#　axisとは1行ごとに処理するという意味
    #　補完された建築年を使って、改めて築年数を計算
    df["築年数"] = datetime.datetime.now().year - df["建築年_西暦"]
    
   #%% 
    # 最寄駅：距離（分）処理
    #　数値だけでない表記を数値に変換するための辞書（マップ）
    replace_dict = {
        "1H?1H30": 75, "1H30?2H": 105, "2H?": 135, "30分?60分": 45}
    #　114行の辞書使って置換。全て文字列型（str）にいったん変換
    df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace(replace_dict).astype(str)
    #　数値型に変換。変換できない値はNaN（欠損）になる
    df["最寄駅：距離（分）"] = pd.to_numeric(df["最寄駅：距離（分）"], errors="coerce")
    
    # 駅ごとの距離平均で補完
    #　駅ごとの距離で平均値を計算
    mean_distance_by_station = df.groupby("最寄駅：名称")["最寄駅：距離（分）"].mean()
    
    #　距離がNaNなら最寄駅の平均を使う。駅名がなければ全体の中央値で補う
    def fill_na_distance(row):
        if pd.isnull(row["最寄駅：距離（分）"]):
            return mean_distance_by_station.get(row["最寄駅：名称"], df["最寄駅：距離（分）"].median())
        return row["最寄駅：距離（分）"]
    #　125行で作成した関数を全行に適用して補完
    df["最寄駅：距離（分）"] = df.apply(fill_na_distance, axis=1)
    
    #建ぺい率/容積率の欠損値も最寄駅の平均値で補完
    for col in ["建ぺい率（％）", "容積率（％）"]:
        mean_by_station = df.groupby("最寄駅：名称")[col].mean()
        overrall_median = df[col].median()
        
        def fill_rate(row):
            if pd.isnull(row[col]):
                return mean_by_station.get(row["最寄駅：名称"], overrall_median)
            return row[col]
        df[col] = df.apply(fill_rate, axis=1)
    
    #　ここまでが主な数値系特微量（面積、建築年、駅距離）の変換と補完
    #%%
    # 都道府県名をダミー変数（サンプル数の多い都道府県に絞る）
    selected_prefectures = ["東京都", "神奈川県", "大阪府", "兵庫県", "埼玉県", "京都府",
                            "愛知県", "北海道", "千葉県"]
    #　指定した都道府県名をダミー変数化
    for pref in selected_prefectures:
        df[f'pref_{pref}'] = (df["都道府県名"] == pref).astype(int)
    #　元の都道府県名は文字列なので削除
    df = df.drop(columns=["都道府県名"])

    # 都市計画をダミー変数
    selected_plans = ["商業地域", "第１種中高層住居専用地域", "第１種住居地域", "準工業地域", "近隣商業地域"]
    for plan in selected_plans:
        df[f"plan_{plan}"] = (df["都市計画"] == plan).astype(int)
    
    #%%
    # 取引の事情等/改装/用途をフラグ変換。通常以外の取引があれば1、なければ0
    df["取引の事情等_事情あり"] = df["取引の事情等"].apply(lambda x: 1 if x != "通常" and pd.notna(x) else 0)
    df["改装済フラグ"] = df["改装"].apply(lambda x: 1 if x == "改装済" else 0)
    df["用途_住宅フラグ"] = df["用途"].apply(lambda x: 1 if pd.notna(x) and "住宅" in x else 0)
    df["今後の利用目的_店舗フラグ"] = df["今後の利用目的"].apply(lambda x: 1 if x == "店舗" else 0)

    #　38行で作成したリスト列をまとめて削除
    df = df.drop(columns=drop_columns, errors="ignore")
    
    #%%
    #　交互作用特徴量を追加
    # 建ぺい率 × 容積率（すでにfloat化済みとして計算）
    if "建ぺい率（％）" in df.columns and "容積率（％）" in df.columns:
        df["建ぺい率_容積率"] = df["建ぺい率（％）"] * df["容積率（％）"]
    
    if "面積（㎡）" in df.columns and "人口密度（k㎡）" in df.columns:
        df["駅距離_人口密度"] = df["面積（㎡）"] * df["人口密度（k㎡）"]
        
    if "最寄駅：距離（分）" in df.columns and "人口密度（k㎡）" in df.columns:
        df["面積_人口密度"] = df["最寄駅：距離（分）"] * df["人口密度（k㎡）"]    
    
    if "築年数" in df.columns and "改装" in df.columns:
        df["築年数_x_改装済"] = df["築年数"] * df["改装済フラグ"]
    
    
    #　前処理が終わったデータを返す
    return df

#%%
# =========================
# 学習・予測処理 (RandomForest)
# =========================

# trainデータ前処理（38行の関数を使用）。copyとしているのは元のデータを壊さないようにするため
#　is_train=Trueを渡すことで、学習用データに必要な前処理をおこなう
df_train = preprocess_data(train_all_data.copy(), is_train=True)
#　使用しない建築年（和暦）と建築年_西暦列を削除。※築年数という特微量を作成したので不要。
df_train = df_train.drop(columns=["建築年_西暦", "建築年", "用途", "今後の利用目的", "改装"], errors="ignore")


# テストデータ前処理

# #　後で予測結果と一緒に出力するためにIDを別で保存
test_ids = test_df["ID"].copy()
#　テストデータにもデータ前処理（38行の関数を使用）
test_df = preprocess_data(test_df.copy(), is_train=False)
#　学習時と同じように不要な列を削除
test_df = test_df.drop(columns=["建築年_西暦", "建築年"], errors="ignore")

#　X_trainの列構成に合わせて、X_testを同じ形に並び変え
#　足りない列があれば0で補完
test_df2 = test_df.reindex(columns=df_train.columns, fill_value=0)

# データ型調整と欠損値補完
#　全ての列を数値型に変換（文字列が混ざっていてもエラーを出さずにNaNに返す）
for col in df_train.columns:
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
    test_df2[col] = pd.to_numeric(test_df2[col], errors="coerce")
    
#　数値型を全てfloat（少数）に変換
numeric_cols = df_train.select_dtypes(include=np.number).columns
df_train[numeric_cols] = df_train[numeric_cols].astype(float)
test_df2[numeric_cols] = test_df2[numeric_cols].astype(float)

# 学習データの平均値で補完（test も同じ値で補完）
mean_values = df_train[numeric_cols].mean()
df_train[numeric_cols] = df_train[numeric_cols].fillna(mean_values)
test_df2[numeric_cols] = test_df2[numeric_cols].fillna(mean_values)

#　X_train：説明変数"取引価格（総額）_log以外
X = df_train.drop(columns=["取引価格（総額）_log", "ID"])
#　y_train：目的変数
y = df_train["取引価格（総額）_log"]

#　X_testから取引価格総額_logを削除
test_df2 = test_df2.drop(columns=["取引価格（総額）_log", "ID"])



#%%
# 訓練データとテストデータに分割 (検証データは不要なため削除)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index) # インデックスを揃える

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

X_train['const'] = 1  # 定数項の列をX_trainに追加
model = sm.OLS(y_train, X_train)

# statsmodels の OLS（最小二乗法）で回帰モデルを構築
model = sm.OLS(y_train, sm.add_constant(X_train))  # 説明変数に定数項を追加
results = model.fit()  # モデルを学習

# モデルのサマリーを表示
print(results.summary())

# 学習時の説明変数の列を保存（例: X.columns）
X_train_cols = X.columns

# test_df2 も X_train と同じスケーラーで変換
test_df2_scaled = scaler.transform(test_df2)
test_df2_scaled = pd.DataFrame(test_df2_scaled, columns=test_df2.columns)

#　定数項を追加
test_df2_const = sm.add_constant(test_df2_scaled)

# 念のため列の順番をX_trainと合わせる
test_df2_const = test_df2_const[X_train.columns]

#　予測
preds_log = results.predict(test_df2_const)

# print(X_train.columns)
# print(test_df2.columns)
# print(test_df2.shape)
# print(preds_log.describe())
# print(preds_log.head(10))
# 結果出力
submission = pd.DataFrame({"ID": test_ids, "取引価格（総額）_log": preds_log})
submission.to_csv("sample_submission_v31_ols.csv", index=False, encoding="utf-8")

print("✅ sample_submission_v31_ols.csv が作成されました！")

#%%グラフの可視化を実行。
# print(submission.head(10))
# # 係数の絶対値が大きい順にソートして表示する
# coef_df = pd.DataFrame({"変数名": results.params.index,"係数": results.params.values})

# # 係数の絶対値の大きい順にソート
# coef_df["係数の絶対値"] = coef_df["係数"].abs()
# sorted_coef_df = coef_df.sort_values(by="係数の絶対値", ascending=False)

# # constを除いてソート・表示
# sorted_coef_df_no_const = sorted_coef_df[sorted_coef_df["変数名"] != "const"]
# print(sorted_coef_df_no_const.head(20))

# # グラフで可視化する
# plt.figure(figsize=(10, 8))
# plt.barh(sorted_coef_df_no_const["変数名"], sorted_coef_df_no_const["係数の絶対値"])
# plt.xlabel("係数の絶対値")
# plt.ylabel("変数名")
# plt.title("回帰係数の絶対値（上位２０変数）")
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

#%%

vif_df = pd.DataFrame()
vif_df["変数名"] = X_train.columns
vif_df["VIF値"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# VIFを大きい順に並べる
vif_df = vif_df.sort_values("VIF値", ascending=False)
print(vif_df)