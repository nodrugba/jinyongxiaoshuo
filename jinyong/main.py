import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import jieba
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import re
import pickle
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud库未安装，将不会生成词云图")

# 设置中文字体支持
def set_chinese_font():
    try:
        # 尝试设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun', 'Arial Unicode MS']  # 首选微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 测试字体是否可用
        font_test = FontProperties(family='SimHei')
        if font_test.get_name() == 'sans':
            print("警告：无法找到中文字体，将尝试使用系统默认字体")
            # 如果找不到中文字体，尝试使用系统默认字体
            import matplotlib.font_manager as fm
            fonts = [f.name for f in fm.fontManager.ttflist]
            for font in fonts:
                if any(c in font for c in ('黑体', '雅黑', '宋体', 'Hei', 'Yahei', 'Song')):
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    print(f"使用系统字体: {font}")
                    break
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将尝试使用特殊方法渲染中文...")

# 合并所有小说内容到一个文件
def merge_novels():
    print("合并小说文件...")
    all_content = ""
    novel_dir = "jinyong"
    all_files = os.listdir(novel_dir)
    
    for file in all_files:
        file_path = os.path.join(novel_dir, file)
        if os.path.isfile(file_path) and file.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_content += content + "\n"
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                        all_content += content + "\n"
                except:
                    print(f"无法读取文件: {file}")
    
    with open("all.txt", "w", encoding='utf-8') as f:
        f.write(all_content)
    
    print(f"合并完成，共计{len(all_content)}字符")
    return all_content

# 分词处理
def segment_text(text):
    print("分词处理中...")
    sentences = re.split(r'[。！？.!?]+', text)
    segmented_sentences = []
    
    for sentence in sentences:
        if sentence.strip():
            words = jieba.lcut(sentence.strip())
            if words:
                segmented_sentences.append(' '.join(words))
    
    print(f"分词完成，共有{len(segmented_sentences)}个句子")
    return segmented_sentences

# 使用TF-IDF创建词向量
def train_tfidf(sentences):
    print("训练TF-IDF模型...")
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 保存模型
    with open("tfidf_model.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    print("模型训练完成并保存")
    return vectorizer, tfidf_matrix

# 读取人名文件
def load_names(name_file="name.txt"):
    print("读取人名文件...")
    with open(name_file, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"读取完成，共有{len(names)}个人名")
    return names

# 生成词云图
def generate_wordcloud(names, valid_names):
    if not WORDCLOUD_AVAILABLE:
        print("词云生成失败：缺少WordCloud库")
        return
    
    print("正在生成人名词云...")
    
    # 为词云准备数据：字典格式 {词: 权重}
    name_freq = {}
    for name in valid_names:
        # 简单起见，所有有效名称都给相同权重
        name_freq[name] = 1
    
    try:
        # 尝试使用不同字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STXihei']
        font_path = None
        
        # 在系统中查找可用的中文字体
        import matplotlib.font_manager as fm
        system_fonts = fm.findSystemFonts()
        
        for font in system_fonts:
            try:
                if any(chinese_font in font.lower() for chinese_font in ['simhei', 'yahei', 'simsun', 'msyh']):
                    font_path = font
                    print(f"找到中文字体：{font}")
                    break
            except:
                continue
        
        # 创建词云
        if font_path:
            wordcloud = WordCloud(
                font_path=font_path,
                background_color='white',
                max_words=100,
                width=800,
                height=600
            ).generate_from_frequencies(name_freq)
        else:
            # 尝试不指定字体
            wordcloud = WordCloud(
                background_color='white',
                max_words=100,
                width=800,
                height=600
            ).generate_from_frequencies(name_freq)
        
        # 显示和保存
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('金庸小说人物名称词云')
        plt.savefig('character_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("词云图已保存为character_wordcloud.png")
    except Exception as e:
        print(f"生成词云时出错: {e}")

# 创建多个图表来展示所有人名
def create_multiple_plots(result_2d, valid_names, is_3d=False, result_3d=None):
    print(f"创建多个图表以展示所有{len(valid_names)}个人名...")
    
    # 每张图最多显示多少个点
    points_per_plot = 200
    
    # 计算需要多少张图
    num_plots = (len(valid_names) + points_per_plot - 1) // points_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * points_per_plot
        end_idx = min(start_idx + points_per_plot, len(valid_names))
        
        if is_3d:
            fig = plt.figure(figsize=(15, 12))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(result_3d[start_idx:end_idx, 0], 
                      result_3d[start_idx:end_idx, 1], 
                      result_3d[start_idx:end_idx, 2], 
                      c='red', alpha=0.5)
            
            # 标注所有人名
            for i in range(start_idx, end_idx):
                ax.text(result_3d[i, 0], result_3d[i, 1], result_3d[i, 2], 
                        valid_names[i], fontsize=8)
            
            plt.title(f'金庸小说人物名称的3D词向量分布 (第{plot_idx+1}部分，共{num_plots}部分)')
            filename = f'character_3d_part{plot_idx+1}_of_{num_plots}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            # 保存SVG格式
            svg_filename = f'character_3d_part{plot_idx+1}_of_{num_plots}.svg'
            plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            plt.figure(figsize=(15, 12))
            plt.scatter(result_2d[start_idx:end_idx, 0], 
                       result_2d[start_idx:end_idx, 1], 
                       c='blue', alpha=0.5)
            
            # 标注所有人名
            for i in range(start_idx, end_idx):
                plt.annotate(valid_names[i], 
                            xy=(result_2d[i, 0], result_2d[i, 1]), 
                            fontsize=8)
            
            plt.title(f'金庸小说人物名称的2D词向量分布 (第{plot_idx+1}部分，共{num_plots}部分)')
            filename = f'character_2d_part{plot_idx+1}_of_{num_plots}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
            # 保存SVG格式
            svg_filename = f'character_2d_part{plot_idx+1}_of_{num_plots}.svg'
            plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"已创建{num_plots}张图表，每张图表最多包含{points_per_plot}个人名")

# 应用PCA降维并可视化
def visualize_embeddings(vectorizer, names):
    print("对人名进行向量化...")
    name_vectors = []
    valid_names = []
    
    # 获取词汇列表和索引映射
    vocab = vectorizer.vocabulary_
    
    for name in names:
        # 检查人名是否在词汇表中
        if name in vocab:
            valid_names.append(name)
            # 创建一个全零向量
            name_vector = np.zeros(len(vocab))
            # 设置对应词的位置为1
            name_vector[vocab[name]] = 1
            name_vectors.append(name_vector)
    
    if not valid_names:
        print("没有找到有效的人名向量，请检查模型和人名列表")
        return
    
    print(f"共有{len(valid_names)}个有效人名向量")
    vectors = np.array(name_vectors)
    
    # 生成词云图作为备用可视化方法
    generate_wordcloud(names, valid_names)
    
    # 使用PCA降维到2D
    print("进行2D PCA降维...")
    pca_2d = PCA(n_components=2)
    result_2d = pca_2d.fit_transform(vectors)
    
    # 使用PCA降维到3D
    print("进行3D PCA降维...")
    pca_3d = PCA(n_components=3)
    result_3d = pca_3d.fit_transform(vectors)
    
    # 1. 创建简化版本的图表（只标注少量人名，便于概览）
    # 绘制2D散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(result_2d[:, 0], result_2d[:, 1], c='blue', alpha=0.5)
    
    # 标注部分重要人物的名字
    labeled = min(30, len(valid_names))  # 最多标注30个
    for i in range(labeled):
        plt.annotate(valid_names[i], xy=(result_2d[i, 0], result_2d[i, 1]), fontsize=9)
    
    plt.title('金庸小说人物名称的2D词向量分布 (概览)')
    plt.savefig('character_2d_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('character_2d_overview.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制3D散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result_3d[:, 0], result_3d[:, 1], result_3d[:, 2], c='red', alpha=0.5)
    
    # 标注部分重要人物的名字
    for i in range(labeled):
        ax.text(result_3d[i, 0], result_3d[i, 1], result_3d[i, 2], valid_names[i], fontsize=9)
    
    plt.title('金庸小说人物名称的3D词向量分布 (概览)')
    plt.savefig('character_3d_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('character_3d_overview.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建多个图表，确保标注所有人名
    create_multiple_plots(result_2d, valid_names, is_3d=False)
    create_multiple_plots(result_2d, valid_names, is_3d=True, result_3d=result_3d)
    
    # 3. 保存完整的散点图但不标注，同时保存人名与坐标的映射数据
    # 2D散点图 - 无标注
    plt.figure(figsize=(12, 10))
    plt.scatter(result_2d[:, 0], result_2d[:, 1], c='blue', alpha=0.5)
    plt.title('金庸小说人物名称的2D词向量分布 (无标注)')
    plt.savefig('character_2d_no_labels.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3D散点图 - 无标注
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result_3d[:, 0], result_3d[:, 1], result_3d[:, 2], c='red', alpha=0.5)
    plt.title('金庸小说人物名称的3D词向量分布 (无标注)')
    plt.savefig('character_3d_no_labels.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存人名与坐标的映射数据
    name_coords_2d = {valid_names[i]: result_2d[i].tolist() for i in range(len(valid_names))}
    name_coords_3d = {valid_names[i]: result_3d[i].tolist() for i in range(len(valid_names))}
    
    with open('name_coords_2d.pkl', 'wb') as f:
        pickle.dump(name_coords_2d, f)
    
    with open('name_coords_3d.pkl', 'wb') as f:
        pickle.dump(name_coords_3d, f)
    
    # 生成CSV文件
    df_2d = pd.DataFrame({
        '人名': valid_names,
        'x坐标': result_2d[:, 0],
        'y坐标': result_2d[:, 1]
    })
    df_2d.to_csv('name_coords_2d.csv', index=False, encoding='utf-8-sig')
    
    df_3d = pd.DataFrame({
        '人名': valid_names,
        'x坐标': result_3d[:, 0],
        'y坐标': result_3d[:, 1],
        'z坐标': result_3d[:, 2]
    })
    df_3d.to_csv('name_coords_3d.csv', index=False, encoding='utf-8-sig')
    
    print("可视化完成，已生成以下文件：")
    print("- 概览图: character_2d_overview.png, character_3d_overview.png")
    print("- 分部分图: character_2d_part1_of_X.png, character_3d_part1_of_X.png 等")
    print("- 无标注图: character_2d_no_labels.png, character_3d_no_labels.png")
    print("- 人名坐标数据: name_coords_2d.pkl, name_coords_3d.pkl")
    print("- CSV文件: name_coords_2d.csv, name_coords_3d.csv")

# 主程序
def main():
    # 设置中文字体
    set_chinese_font()
    
    # 合并所有小说内容
    if not os.path.exists("all.txt"):
        text = merge_novels()
    else:
        print("检测到all.txt已存在，跳过合并步骤")
        with open("all.txt", 'r', encoding='utf-8') as f:
            text = f.read()
    
    # 分词处理
    sentences = segment_text(text)
    
    # 训练模型
    if not os.path.exists("tfidf_model.pkl"):
        vectorizer, _ = train_tfidf(sentences)
    else:
        print("检测到模型已存在，直接加载")
        with open("tfidf_model.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    
    # 读取人名
    names = load_names()
    
    # 可视化
    visualize_embeddings(vectorizer, names)
    
    print("所有处理完成！")

if __name__ == "__main__":
    main() 