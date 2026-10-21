#!/usr/bin/env python3
"""
使用 python-docx 生成完整的课程考核报告
按照《计算机学报》模版格式
"""

from docx import Document
from docx.shared import Pt, Cm, Inches, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

doc = Document()

# ============================
# 页面设置 A4
# ============================
for section in doc.sections:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

# ============================
# 样式设置
# ============================
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
pf = style.paragraph_format
pf.line_spacing = Pt(16)
pf.space_before = Pt(0)
pf.space_after = Pt(0)

def set_run_font(run, cn_font='宋体', en_font='Times New Roman', size=Pt(10.5), bold=False, italic=False, color=None):
    """设置run的中英文字体"""
    run.font.name = en_font
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.element.rPr.rFonts.set(qn('w:eastAsia'), cn_font)
    if color:
        run.font.color.rgb = color

def add_paragraph_with_font(doc, text, cn_font='宋体', en_font='Times New Roman', 
                            size=Pt(10.5), bold=False, alignment=None, 
                            space_before=Pt(0), space_after=Pt(0),
                            first_line_indent=None, line_spacing=Pt(16)):
    """添加段落并设置字体"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = space_before
    pf.space_after = space_after
    pf.line_spacing = line_spacing
    if alignment is not None:
        pf.alignment = alignment
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    run = p.add_run(text)
    set_run_font(run, cn_font, en_font, size, bold)
    return p

def add_mixed_para(doc, parts, alignment=None, first_line_indent=None, 
                   space_before=Pt(0), space_after=Pt(0), line_spacing=Pt(16)):
    """添加混合字体段落
    parts: list of (text, cn_font, en_font, size, bold, italic)
    """
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = space_before
    pf.space_after = space_after
    pf.line_spacing = line_spacing
    if alignment is not None:
        pf.alignment = alignment
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    for part in parts:
        text = part[0]
        cn_font = part[1] if len(part) > 1 else '宋体'
        en_font = part[2] if len(part) > 2 else 'Times New Roman'
        size = part[3] if len(part) > 3 else Pt(10.5)
        bold = part[4] if len(part) > 4 else False
        italic = part[5] if len(part) > 5 else False
        run = p.add_run(text)
        set_run_font(run, cn_font, en_font, size, bold, italic)
    return p

def section_heading(doc, text):
    """一级标题: 黑体 小四"""
    p = add_paragraph_with_font(doc, text, cn_font='黑体', en_font='Arial', 
                                 size=Pt(12), bold=True,
                                 space_before=Pt(12), space_after=Pt(6),
                                 line_spacing=Pt(18))
    return p

def sub_heading(doc, text):
    """二级标题: 黑体 五号"""
    p = add_paragraph_with_font(doc, text, cn_font='黑体', en_font='Arial',
                                 size=Pt(11), bold=True,
                                 space_before=Pt(8), space_after=Pt(4),
                                 line_spacing=Pt(16))
    return p

def body_text(doc, text, indent=True):
    """正文段落: 宋体 五号, 首行缩进"""
    p = add_paragraph_with_font(doc, text, cn_font='宋体', en_font='Times New Roman',
                                 size=Pt(10.5), 
                                 first_line_indent=Cm(0.74) if indent else None,
                                 line_spacing=Pt(16),
                                 alignment=WD_ALIGN_PARAGRAPH.JUSTIFY)
    return p

def body_mixed(doc, parts, indent=True):
    """正文混合段落"""
    return add_mixed_para(doc, parts,
                          first_line_indent=Cm(0.74) if indent else None,
                          line_spacing=Pt(16),
                          alignment=WD_ALIGN_PARAGRAPH.JUSTIFY)

def add_figure(doc, img_path, width_cm, caption_cn, caption_en):
    """添加图片及中英文图注"""
    # 图片段落
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(img_path, width=Cm(width_cm))
    
    # 中文图注
    p2 = add_paragraph_with_font(doc, caption_cn, cn_font='宋体', en_font='Times New Roman',
                                  size=Pt(9), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                                  space_before=Pt(2), space_after=Pt(0),
                                  line_spacing=Pt(14))
    # 英文图注
    p3 = add_paragraph_with_font(doc, caption_en, cn_font='宋体', en_font='Times New Roman',
                                  size=Pt(8), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                                  space_before=Pt(0), space_after=Pt(6),
                                  line_spacing=Pt(13))
    return p

def ref_item(doc, text):
    """参考文献条目"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = Pt(13)
    pf.first_line_indent = Cm(-0.74)
    pf.left_indent = Cm(0.74)
    run = p.add_run(text)
    set_run_font(run, '宋体', 'Times New Roman', Pt(9))
    return p


# ================================================================
#                    开始构建文档
# ================================================================

# ===== 期刊名头 =====
add_paragraph_with_font(doc, '计  算  机  学  报', cn_font='宋体', en_font='Times New Roman',
                        size=Pt(16), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(2))

add_paragraph_with_font(doc, 'CHINESE JOURNAL OF COMPUTERS', cn_font='Times New Roman', en_font='Times New Roman',
                        size=Pt(9), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(4))

# 分隔线
p_line = doc.add_paragraph()
p_line.paragraph_format.space_after = Pt(8)
pPr = p_line._p.get_or_add_pPr()
pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="6" w:space="1" w:color="000000"/></w:pBdr>')
pPr.append(pBdr)

# ===== 中文标题 =====
add_paragraph_with_font(doc, '基于K-近邻方法的模式分类研究与分析',
                        cn_font='黑体', en_font='Arial',
                        size=Pt(18), bold=True,
                        alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=Pt(16), space_after=Pt(10))

# ===== 作者 =====
add_paragraph_with_font(doc, 'xxx',
                        cn_font='楷体', en_font='Times New Roman',
                        size=Pt(12), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(4))

add_paragraph_with_font(doc, '(苏州大学计算机科学与技术学院  苏州  215006)',
                        cn_font='宋体', en_font='Times New Roman',
                        size=Pt(9), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(8))

# ===== 中文摘要 =====
p = doc.add_paragraph()
pf = p.paragraph_format
pf.first_line_indent = Cm(0.74)
pf.line_spacing = Pt(16)
pf.space_before = Pt(4)
pf.space_after = Pt(2)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = p.add_run('摘  要')
set_run_font(r1, '黑体', 'Arial', Pt(10.5), bold=True)
r2 = p.add_run('  K-近邻（KNN）方法是模式识别领域中最基本且应用最广泛的非参数分类方法之一。本文系统研究了KNN分类算法的基本原理及其关键影响因素。首先，阐述了KNN算法的基本思想、距离度量方式和K值选取策略；其次，深入分析了加权KNN方法的改进机理。在实验部分，以Iris数据集为对象，利用Python编程实现KNN分类器，通过10折交叉验证系统地研究了K值、距离度量方式和加权策略对分类性能的影响，并与贝叶斯分类器、Fisher线性判别分析、支持向量机和决策树等方法进行了对比分析。实验结果表明：KNN方法在合理选择K值和距离度量的条件下，能够取得较高的分类准确率；加权KNN方法在K值较大时具有更稳定的性能表现。本文的研究为理解和应用KNN方法提供了系统的实验参考。')
set_run_font(r2, '宋体', 'Times New Roman', Pt(10.5))

# 关键词
p = doc.add_paragraph()
pf = p.paragraph_format
pf.first_line_indent = Cm(0.74)
pf.line_spacing = Pt(16)
pf.space_after = Pt(2)
r1 = p.add_run('关键词')
set_run_font(r1, '黑体', 'Arial', Pt(10.5), bold=True)
r2 = p.add_run('  K-近邻方法；模式分类；距离度量；交叉验证；加权KNN')
set_run_font(r2, '宋体', 'Times New Roman', Pt(10.5))

# 中图法分类号
p = doc.add_paragraph()
pf = p.paragraph_format
pf.first_line_indent = Cm(0.74)
pf.line_spacing = Pt(16)
pf.space_after = Pt(6)
r1 = p.add_run('中图法分类号')
set_run_font(r1, '黑体', 'Arial', Pt(10.5), bold=True)
r2 = p.add_run('  TP391')
set_run_font(r2, '宋体', 'Times New Roman', Pt(10.5))

# ===== 英文标题 =====
add_paragraph_with_font(doc, 'Research and Analysis of Pattern Classification Based on K-Nearest Neighbor Method',
                        en_font='Times New Roman', cn_font='Times New Roman',
                        size=Pt(14), bold=True,
                        alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=Pt(8), space_after=Pt(4))

add_paragraph_with_font(doc, 'XXX',
                        en_font='Times New Roman', cn_font='Times New Roman',
                        size=Pt(10.5), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(2))

p = add_paragraph_with_font(doc, '(School of Computer Science and Technology, Soochow University, Suzhou 215006)',
                        en_font='Times New Roman', cn_font='Times New Roman',
                        size=Pt(9), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(4))
p.runs[0].font.italic = True

# 英文摘要
p = doc.add_paragraph()
pf = p.paragraph_format
pf.first_line_indent = Cm(0.74)
pf.line_spacing = Pt(15)
pf.space_before = Pt(2)
pf.space_after = Pt(2)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = p.add_run('Abstract')
set_run_font(r1, 'Times New Roman', 'Times New Roman', Pt(10), bold=True)
r2 = p.add_run('  The K-Nearest Neighbor (KNN) method is one of the most fundamental and widely used non-parametric classification methods in pattern recognition. This paper systematically studies the basic principles of KNN and its key influencing factors. Firstly, the basic idea, distance metrics, and K-value selection strategies are elaborated. Secondly, the improvement mechanism of the weighted KNN method is analyzed in depth. In the experimental part, using the Iris dataset, a KNN classifier is implemented with Python, and the effects of K-value, distance metric, and weighting strategy on classification performance are systematically investigated through 10-fold cross-validation. Comparative analyses with Bayesian classifier, Fisher LDA, SVM, and decision tree are also conducted. Results demonstrate that KNN can achieve high classification accuracy with proper K-value and distance metric selection, and the weighted KNN exhibits more stable performance for larger K values.')
set_run_font(r2, 'Times New Roman', 'Times New Roman', Pt(10))

# 英文关键词
p = doc.add_paragraph()
pf = p.paragraph_format
pf.first_line_indent = Cm(0.74)
pf.line_spacing = Pt(15)
pf.space_after = Pt(4)
r1 = p.add_run('Keywords')
set_run_font(r1, 'Times New Roman', 'Times New Roman', Pt(10), bold=True)
r2 = p.add_run('  K-nearest neighbor; pattern classification; distance metric; cross-validation; weighted KNN')
set_run_font(r2, 'Times New Roman', 'Times New Roman', Pt(10))

# 分隔线
p_line2 = doc.add_paragraph()
p_line2.paragraph_format.space_after = Pt(4)
pPr2 = p_line2._p.get_or_add_pPr()
pBdr2 = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="4" w:space="1" w:color="000000"/></w:pBdr>')
pPr2.append(pBdr2)

# ================================================================
#                       正  文
# ================================================================

# ===== 1 引言 =====
section_heading(doc, '1  引言')

body_text(doc, '模式识别是人工智能和机器学习领域的重要分支，其核心任务是根据已有样本的特征信息建立分类规则，从而实现对未知样本的自动分类[1]。在模式识别系统中，分类器的设计直接决定了识别系统的性能。经过数十年的发展，研究者提出了多种经典的分类方法，包括基于统计决策的贝叶斯方法、基于判别函数的Fisher线性判别法、基于实例的K-近邻法等[2]。')

body_text(doc, 'K-近邻（K-Nearest Neighbor, KNN）方法由Cover和Hart于1967年提出[3]，是一种基于实例的学习（Instance-based Learning）算法。与贝叶斯分类器和Fisher判别分析等参数方法不同，KNN方法不需要对数据的概率分布作任何假设，也不需要通过训练过程建立显式的分类模型，因此被称为"懒惰学习"（Lazy Learning）方法。其基本思想十分直观：对于一个待分类样本，在特征空间中寻找与其距离最近的K个训练样本，根据这K个近邻的类别信息，通过多数表决规则决定待分类样本的类别。')

body_text(doc, '尽管KNN方法的思想简单，但其分类性能受多个因素的影响，主要包括：（1）K值的选取——K值过小容易导致过拟合，K值过大则可能引入大量噪声样本；（2）距离度量的选择——不同的距离度量会导致不同的近邻集合，从而影响分类结果；（3）权重策略——是否根据距离对近邻的投票进行加权。这些因素的合理选择对KNN分类器的性能至关重要[4]。如何在具体应用中合理设定这些参数，是KNN方法研究的核心问题。')

body_text(doc, '本文以模式识别课程中的K-近邻方法为核心，系统地研究了KNN分类器的基本原理和关键影响因素。通过在经典Iris数据集上的实验，深入分析了K值、距离度量和加权策略对分类性能的影响，并与其他经典分类方法进行了对比，为理解和应用KNN方法提供了翔实的实验参考。')

# ===== 2 方法论述 =====
section_heading(doc, '2  方法论述')
sub_heading(doc, '2.1  KNN算法基本原理')

body_text(doc, '设训练样本集为T = {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}，其中xᵢ为第i个样本的n维特征向量，yᵢ∈{c₁, c₂, ..., cₘ}为其对应的类别标签。对于一个待分类的新样本x，KNN算法的基本步骤如下：')

body_text(doc, '（1）计算待分类样本x与训练集中所有N个样本之间的距离d(x, xᵢ)，i = 1, 2, ..., N。')

body_text(doc, '（2）将这N个距离按从小到大排序，选取距离最近的K个训练样本，构成x的K-近邻集合Nₖ(x)。')

body_text(doc, '（3）根据多数表决规则，将x判定为K-近邻集合中出现频率最高的类别，即分类决策函数为：')

add_paragraph_with_font(doc, 'y = arg max  Σ I(yᵢ = cⱼ) ,   xᵢ ∈ Nₖ(x)',
                        cn_font='宋体', en_font='Times New Roman',
                        size=Pt(10.5), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=Pt(4), space_after=Pt(4),
                        line_spacing=Pt(18))

body_text(doc, '其中I(·)为指示函数，当条件成立时取值1，否则取值0。上式的含义是：在K个近邻中，统计每个类别出现的次数，将出现次数最多的类别作为分类结果。')

body_text(doc, 'Cover和Hart在1967年的经典工作中证明了一个重要的理论结果：当训练样本数量N趋于无穷时，最近邻分类器（K=1）的渐近错误率R*满足R* ≤ R(1+R)/(M-1+R)，其中R为贝叶斯最优错误率，M为类别数。特别地，当M=2时，R* ≤ 2R(1-R) ≤ 2R。这一结论表明最近邻方法虽然简单，但具有坚实的统计理论基础[3]。')

sub_heading(doc, '2.2  距离度量方法')

body_text(doc, '距离度量是KNN算法的核心要素之一。不同的距离度量方式反映了特征空间中样本间不同的相似性概念[5]。对于n维特征空间中的两个样本x = (x₁, x₂, ..., xₙ)和z = (z₁, z₂, ..., zₙ)，常用的距离度量方法包括：')

body_text(doc, '（1）欧氏距离（Euclidean Distance）：d(x,z) = √[Σᵢ(xᵢ-zᵢ)²]。欧氏距离是最直观、最常用的距离度量，它衡量的是两点之间的直线距离。在各特征具有相同重要性和相似量纲时，欧氏距离是一个合理的选择。')

body_text(doc, '（2）曼哈顿距离（Manhattan Distance）：d(x,z) = Σᵢ|xᵢ-zᵢ|。又称城市街区距离或L1距离，它衡量的是沿坐标轴方向的距离之和。在特征空间中存在较多离群值时，曼哈顿距离比欧氏距离更加鲁棒。')

body_text(doc, '（3）切比雪夫距离（Chebyshev Distance）：d(x,z) = maxᵢ|xᵢ-zᵢ|。它取各维度差值绝对值的最大值作为距离。切比雪夫距离仅关注差异最大的那个维度，适用于需要重点关注最大特征差异的场景。')

body_text(doc, '（4）闵可夫斯基距离（Minkowski Distance）：d(x,z) = [Σᵢ|xᵢ-zᵢ|ᵖ]^(1/p)。它是上述距离的统一形式，参数p控制距离的计算方式。当p=1时即为曼哈顿距离，p=2时为欧氏距离，p→∞时退化为切比雪夫距离。通过调节p值，可以灵活地改变距离度量对不同维度差异的敏感程度。')

body_text(doc, '需要特别指出的是，在应用KNN算法之前，通常需要对所有特征进行标准化处理（如零均值单位方差标准化），以消除不同特征量纲差异对距离计算的影响。若不进行标准化，取值范围较大的特征将主导距离的计算，导致其他特征的信息被忽略。')

sub_heading(doc, '2.3  K值选择策略')

body_text(doc, 'K值是KNN算法中最重要的超参数，它直接影响分类器的偏差-方差权衡（Bias-Variance Tradeoff）：')

body_text(doc, '当K值较小时（如K=1），分类决策仅依赖于距离最近的少数几个样本，模型的复杂度较高，具有较低的偏差但较高的方差。这意味着分类器能够精确地拟合训练数据中的细节和噪声，但对于新的测试样本泛化能力不足，容易发生过拟合。极端情况下，K=1时待分类样本的类别完全由距离最近的单个样本决定，分类边界极其复杂。')

body_text(doc, '当K值较大时，分类决策综合了更多训练样本的信息，模型的平滑度更高，具有较高的偏差但较低的方差。分类边界趋于平滑，抗噪声能力增强。但K值过大可能将大量不属于目标类别的远距离样本纳入决策范围，导致欠拟合。极端情况下，当K等于训练样本总数N时，所有待分类样本都将被判定为训练集中出现频率最高的类别。')

body_text(doc, '在实际应用中，K值通常通过交叉验证（Cross-Validation）方法来选取[8]。最常用的是k折交叉验证：将训练集随机分为k个大小近似的互不相交的子集，轮流将其中一个子集作为验证集、其余k-1个子集作为训练集，在每种划分下评估不同K值对应的分类准确率，最终选择使平均准确率最高的K值。此外，实践中K值一般选取奇数，以避免在二分类问题中出现票数相同的情况。')

sub_heading(doc, '2.4  加权KNN方法')

body_text(doc, '传统KNN方法中，K个近邻对分类决策具有相同的投票权重。然而这种等权投票策略存在一个不合理之处——距离较近的邻居通常比距离较远的邻居更能反映待分类样本的类别信息。Dudani于1976年提出了距离加权KNN方法[6]，根据距离赋予不同的投票权重，使距离越近的近邻权重越大。')

body_text(doc, '最常用的加权方式是距离倒数加权法，即将第i个近邻的投票权重设为wᵢ = 1/d(x, xᵢ)。加权KNN的分类决策函数可表示为：')

add_paragraph_with_font(doc, 'y = arg max  Σ wᵢ · I(yᵢ = cⱼ) ,   xᵢ ∈ Nₖ(x)',
                        cn_font='宋体', en_font='Times New Roman',
                        size=Pt(10.5), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=Pt(4), space_after=Pt(4),
                        line_spacing=Pt(18))

body_text(doc, '加权KNN方法在K值选择方面具有更好的鲁棒性：即使K值选取较大，由于距离较远的样本权重较低，其对分类决策的影响也相对较小，从而在一定程度上缓解了K值过大引起的欠拟合问题。同时，加权策略也降低了分类结果对K值精确选择的敏感性，使得分类器在更大的K值范围内保持稳定的性能。')

# ===== 3 实验与分析 =====
section_heading(doc, '3  实验与分析')
sub_heading(doc, '3.1  实验数据集与设置')

body_text(doc, '本文选用经典的Iris（鸢尾花）数据集作为实验对象。该数据集由Fisher于1936年提出[7]，是模式识别领域最常用的基准数据集之一。Iris数据集包含150个样本，分为3个类别（Setosa、Versicolour、Virginica），每类各50个样本。每个样本具有4个特征：花萼长度（sepal length）、花萼宽度（sepal width）、花瓣长度（petal length）和花瓣宽度（petal width），均为连续型特征。')

body_text(doc, '实验环境为Python 3.10，使用scikit-learn机器学习库实现各分类算法。所有实验均采用10折分层交叉验证（10-fold Stratified Cross-Validation）进行性能评估，以确保每折中各类别样本的比例与整体数据集一致。在所有实验开始前，对全部特征进行零均值单位方差标准化处理，以消除不同特征量纲差异对距离计算的影响。评价指标采用分类准确率（Accuracy），即正确分类的样本数占总样本数的比例。')

sub_heading(doc, '3.2  K值对分类性能的影响')

body_text(doc, '为系统考察K值对KNN分类准确率的影响，我们在K=1至K=30的范围内逐一进行了实验。首先，图1展示了在K=1、K=5和K=15三个代表性K值下，KNN分类器在PCA降维后的二维空间中的决策边界变化情况。')

# 图 1
add_figure(doc, '/home/claude/fig1.png', 15.0,
           '图1  不同K值下KNN分类器的决策边界（PCA降维至二维）',
           'Fig.1  Decision boundaries of KNN classifier with different K values (PCA reduced to 2D)')

body_text(doc, '从图1可以直观地观察到：当K=1时，决策边界极其复杂且不规则，呈现典型的过拟合特征，分类器对训练样本中的每个点都做出了精确拟合；当K=5时，决策边界变得较为平滑，分类区域更加规整，在拟合能力和泛化能力之间取得了较好的平衡；当K=15时，决策边界进一步平滑化，分类区域趋于简单的大区域划分，可能忽略了一些细节的类别边界信息。')

body_text(doc, '图2给出了K值从1到30变化时，10折交叉验证平均准确率的变化曲线（蓝色实线），同时展示了标准差范围（蓝色阴影区域）。可以观察到以下规律：')

# 图 2
add_figure(doc, '/home/claude/fig2.png', 13.0,
           '图2  K值对KNN分类准确率的影响（10折交叉验证）',
           'Fig.2  Effect of K value on KNN classification accuracy (10-fold CV)')

body_text(doc, '（1）当K值较小（K<5）时，分类准确率相对较低且标准差较大，反映了小K值下模型对局部噪声的敏感性和较高的方差。')

body_text(doc, '（2）在K=5至K=15的范围内，准确率维持在较高水平且相对稳定，说明适中的K值能够在偏差和方差之间取得良好的平衡。实验中K=8时取得了最高的平均准确率（96.67%）。')

body_text(doc, '（3）当K值继续增大（K>20）时，准确率出现缓慢下降的趋势，表明过多的远距离样本参与投票降低了分类器的判别能力，开始出现欠拟合迹象。')

sub_heading(doc, '3.3  距离度量对分类结果的影响')

body_text(doc, '为比较不同距离度量对KNN分类性能的影响，我们分别采用欧氏距离（Euclidean）、曼哈顿距离（Manhattan）、切比雪夫距离（Chebyshev）和Minkowski距离（p=3）四种度量方式，在K=1至K=20的范围内进行了系统实验。图3展示了在不同距离度量下K值与分类准确率的关系。')

# 图 3
add_figure(doc, '/home/claude/fig3.png', 13.0,
           '图3  不同距离度量下K值与分类准确率的关系',
           'Fig.3  Relationship between K value and accuracy under different distance metrics')

body_text(doc, '从图3可以得出以下分析：（1）欧氏距离和曼哈顿距离在大多数K值下表现相近且总体优于其他两种距离，说明在经过标准化处理后，L1和L2范数在Iris数据集上具有相似的度量效果；（2）切比雪夫距离的性能波动较大，尤其在K值较小时表现不稳定，这是因为切比雪夫距离仅考虑了单个维度的最大差异，丢失了其他维度的距离信息；（3）Minkowski距离（p=3）的表现介于欧氏距离和切比雪夫距离之间，随着p值增大，距离度量逐渐倾向于关注差异最大的维度；（4）随着K值增大，四种距离度量的准确率趋于接近，说明K值较大时多数投票的平均效应减弱了距离度量选择的影响。')

sub_heading(doc, '3.4  加权KNN与普通KNN的比较')

body_text(doc, '图4展示了等权KNN（Uniform Weight）和距离加权KNN（Distance Weight）在K=1至K=25范围内分类性能的对比结果。')

# 图 4
add_figure(doc, '/home/claude/fig6.png', 13.0,
           '图4  等权KNN与距离加权KNN的性能比较',
           'Fig.4  Performance comparison of uniform and distance-weighted KNN')

body_text(doc, '实验结果显示了两种方法的差异：（1）在K值较小时（K<10），两种方法的分类性能几乎没有差别。这是合理的，因为K较小时近邻样本的距离差异不大，加权与否对投票结果的影响有限；（2）随着K值增大，等权KNN的准确率逐渐下降，而加权KNN的准确率下降速度明显更慢，在K=15至K=25的区间内保持了更好的稳定性。这验证了距离加权策略的有效性：通过赋予距离较远样本较低的权重，减弱了非同类远距离样本对分类决策的不良干扰，从而在一定程度上缓解了K值过大导致的欠拟合问题。')

sub_heading(doc, '3.5  与其他分类方法的比较')

body_text(doc, '为全面评估KNN方法的分类性能，将其与贝叶斯分类器（Gaussian Naive Bayes）、Fisher线性判别分析（Linear Discriminant Analysis, LDA）、支持向量机（SVM, 采用RBF核函数）和决策树（Decision Tree）进行了对比。所有方法均在相同的实验条件下进行10折分层交叉验证，结果如表1所示。')

# 表1 标题
add_paragraph_with_font(doc, '表1  不同分类方法在Iris数据集上的准确率比较',
                        cn_font='黑体', en_font='Times New Roman',
                        size=Pt(9), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_before=Pt(6), space_after=Pt(2))

add_paragraph_with_font(doc, 'Table 1  Accuracy comparison of different classification methods on Iris dataset',
                        en_font='Times New Roman', cn_font='Times New Roman',
                        size=Pt(8), alignment=WD_ALIGN_PARAGRAPH.CENTER,
                        space_after=Pt(4))

# 创建表格
table = doc.add_table(rows=7, cols=3)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.style = 'Table Grid'

# 表头
headers = ['分类方法', '平均准确率', '标准差']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(h)
    set_run_font(r, '黑体', 'Times New Roman', Pt(9), bold=True)

# 数据
data = [
    ['KNN (K=5)', '0.9533', '0.0521'],
    ['KNN (K=9)', '0.9533', '0.0521'],
    ['Bayes', '0.9533', '0.0521'],
    ['Fisher LDA', '0.9733', '0.0442'],
    ['SVM (RBF)', '0.9600', '0.0442'],
    ['Decision Tree', '0.9333', '0.0516'],
]
for row_idx, row_data in enumerate(data):
    for col_idx, val in enumerate(row_data):
        cell = table.rows[row_idx + 1].cells[col_idx]
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(val)
        set_run_font(r, '宋体', 'Times New Roman', Pt(9))

# 设置三线表样式
from docx.oxml import OxmlElement
def set_cell_border(cell, **kwargs):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            element = OxmlElement(f'w:{edge}')
            for attr, val in kwargs[edge].items():
                element.set(qn(f'w:{attr}'), str(val))
            tcBorders.append(element)
    tcPr.append(tcBorders)

# 清除所有边框，只保留三线表格式
for row in table.rows:
    for cell in row.cells:
        set_cell_border(cell,
            top={'val': 'nil'},
            bottom={'val': 'nil'},
            left={'val': 'nil'},
            right={'val': 'nil'})

# 顶线
for cell in table.rows[0].cells:
    set_cell_border(cell, top={'val': 'single', 'sz': '12', 'color': '000000'})
# 表头底线
for cell in table.rows[0].cells:
    set_cell_border(cell, bottom={'val': 'single', 'sz': '6', 'color': '000000'})
# 底线
for cell in table.rows[-1].cells:
    set_cell_border(cell, bottom={'val': 'single', 'sz': '12', 'color': '000000'})

# 表后空行
doc.add_paragraph().paragraph_format.space_after = Pt(2)

body_text(doc, '图5以柱状图的形式直观展示了各方法的分类准确率对比。')

# 图 5
add_figure(doc, '/home/claude/fig4.png', 13.5,
           '图5  不同分类方法的准确率比较（10折交叉验证）',
           'Fig.5  Accuracy comparison of different classification methods (10-fold CV)')

body_text(doc, '从表1和图5中可以得出以下结论：')

body_text(doc, '（1）Fisher线性判别分析取得了最高的平均准确率（97.33%），这与Iris数据集各类别间的线性可分性较好有关。Fisher LDA通过寻找使类间散布最大、类内散布最小的投影方向，能够有效地利用类别间的差异信息，在线性可分或近似线性可分的数据集上具有优势。')

body_text(doc, '（2）KNN（K=5和K=9）、贝叶斯分类器和SVM的分类准确率均在95%以上且非常接近，表明这些方法在处理Iris这一经典数据集时具有可比较的性能。贝叶斯分类器假设特征之间条件独立且服从高斯分布，而Iris数据集的分布特性恰好近似满足这一假设。')

body_text(doc, '（3）决策树的性能相对较低（93.33%），这是因为单一决策树的泛化能力有限，容易受到训练集随机划分方式的影响。在实际应用中，可以通过随机森林（Random Forest）等集成方法来改善决策树的稳定性和准确率。')

body_text(doc, '（4）KNN方法虽然在准确率上略低于Fisher LDA，但其无需假设数据分布形式、实现简单、易于理解的特点使其在许多实际场景中仍然具有独特的应用价值，尤其适用于分布形式未知或高度非线性的分类任务。')

body_text(doc, '图6展示了KNN（K=5）在30%测试集上的混淆矩阵，以更详细地展示分类结果。')

# 图 6
add_figure(doc, '/home/claude/fig5.png', 9.5,
           '图6  KNN（K=5）分类器的混淆矩阵',
           'Fig.6  Confusion matrix of KNN (K=5) classifier')

body_text(doc, '从混淆矩阵可以看出，Setosa类别能够被完美分类（无误判），而Versicolour和Virginica之间存在少量误分类样本。这与两个类别在特征空间中存在一定程度的重叠有关，是Iris数据集的固有特点。在实际应用中，可以考虑通过增加更具判别力的特征或采用特征提取方法（如PCA、LDA投影）来进一步改善这两类之间的分类性能。')

# ===== 4 总结 =====
section_heading(doc, '4  总结')

body_text(doc, '本文以K-近邻方法为研究对象，对其基本原理、距离度量、K值选择和加权策略进行了系统的理论分析和实验研究。通过在Iris数据集上的大量实验，得出以下主要结论：')

body_text(doc, '（1）K值的选择对KNN分类器的性能具有显著影响。K值过小导致过拟合，K值过大导致欠拟合。本文实验中K=8附近取得了最优的分类性能（96.67%）。推荐在实际应用中使用交叉验证方法来系统地确定最优K值。')

body_text(doc, '（2）在四种距离度量的比较中，欧氏距离和曼哈顿距离在Iris数据集上表现稳健，适合作为默认选择。切比雪夫距离由于仅关注单维最大差异，性能波动较大。距离度量的选择应根据具体数据的分布特征来决定。')

body_text(doc, '（3）加权KNN方法通过赋予距离较近的样本更大的投票权重，能够有效提升K值较大时的分类稳定性。实验表明，在K>15的区间内，加权KNN的性能显著优于等权KNN，验证了距离加权策略的有效性。')

body_text(doc, '（4）在与贝叶斯分类器、Fisher LDA、SVM和决策树的对比中，KNN方法取得了具有竞争力的分类性能。尽管Fisher LDA在Iris数据集上略占优势，但KNN方法不依赖数据分布假设的特点使其具有更广泛的适用性。')

body_text(doc, '综上所述，KNN方法作为一种简单而有效的非参数分类方法，在模式识别领域具有重要的理论价值和应用前景。未来的研究方向包括：大规模数据集上的高效近邻搜索算法（如KD-Tree、Ball-Tree）、自适应K值选择方法、特征加权距离度量、以及KNN与深度学习特征提取方法的融合等[9,10]。')

# ===== 参考文献 =====
# 分隔线
p_line3 = doc.add_paragraph()
p_line3.paragraph_format.space_before = Pt(10)
pPr3 = p_line3._p.get_or_add_pPr()
pBdr3 = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="4" w:space="1" w:color="000000"/></w:pBdr>')
pPr3.append(pBdr3)

add_paragraph_with_font(doc, '参  考  文  献',
                        cn_font='黑体', en_font='Arial',
                        size=Pt(11), bold=True,
                        space_before=Pt(6), space_after=Pt(4))

refs = [
    '[1]  孙即祥. 现代模式识别. 长沙: 国防科技大学出版社, 2003.',
    '[2]  杨淑莹. 模式识别与智能计算. 北京: 电子工业出版社, 2015.',
    '[3]  Cover T, Hart P. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 1967, 13(1): 21-27.',
    '[4]  Duda R O, Hart P E, Stork D G. Pattern Classification. 2nd ed. New York: Wiley, 2001.',
    '[5]  Fukunaga K. Introduction to Statistical Pattern Recognition. 2nd ed. San Diego: Academic Press, 1990.',
    '[6]  Dudani S A. The distance-weighted k-nearest-neighbor rule. IEEE Transactions on Systems, Man, and Cybernetics, 1976, 6(4): 325-327.',
    '[7]  Fisher R A. The use of multiple measurements in taxonomic problems. Annals of Eugenics, 1936, 7(2): 179-188.',
    '[8]  Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd ed. New York: Springer, 2009.',
    '[9]  Bishop C M. Pattern Recognition and Machine Learning. New York: Springer, 2006.',
    '[10] Altman N S. An introduction to kernel and nearest-neighbor nonparametric regression. The American Statistician, 1992, 46(3): 175-185.',
]
for r in refs:
    ref_item(doc, r)

# ============================
# 保存文档
# ============================
output_path = '/home/claude/knn_report_final.docx'
doc.save(output_path)
print(f'Document saved to {output_path}')
