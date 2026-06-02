from __future__ import annotations

import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}
W = NS["w"]
XML = "http://www.w3.org/XML/1998/namespace"


SOURCE = Path(r"F:/科研资料/paper_6/手稿/paper-v1.9.docx")
OUTPUT = Path(r"F:/科研资料/paper_6/手稿/paper-v1.9-中文修订-带修改痕迹.docx")
AUTHOR = "Codex"
DATE = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


def paragraph_text(p: etree._Element) -> str:
    chunks: list[str] = []
    for node in p.xpath(".//w:t | .//w:delText", namespaces=NS):
        chunks.append(node.text or "")
    return "".join(chunks)


def next_revision_id(root: etree._Element) -> int:
    ids = []
    for el in root.xpath(".//*[@w:id]", namespaces=NS):
        try:
            ids.append(int(el.get(qn("w:id"))))
        except (TypeError, ValueError):
            pass
    return max(ids, default=0) + 1


def make_run_text(tag: str, text: str) -> etree._Element:
    r = etree.Element(qn("w:r"))
    t = etree.SubElement(r, qn(tag))
    t.set(f"{{{XML}}}space", "preserve")
    t.text = text
    return r


def replace_paragraph_with_revision(
    p: etree._Element,
    new_text: str,
    rev_id: int,
) -> int:
    old_text = paragraph_text(p)
    pPr = p.find(qn("w:pPr"))

    for child in list(p):
        if child is not pPr:
            p.remove(child)

    if old_text:
        deleted = etree.SubElement(p, qn("w:del"))
        deleted.set(qn("w:id"), str(rev_id))
        deleted.set(qn("w:author"), AUTHOR)
        deleted.set(qn("w:date"), DATE)
        deleted.append(make_run_text("w:delText", old_text))
        rev_id += 1

    if new_text:
        inserted = etree.SubElement(p, qn("w:ins"))
        inserted.set(qn("w:id"), str(rev_id))
        inserted.set(qn("w:author"), AUTHOR)
        inserted.set(qn("w:date"), DATE)
        inserted.append(make_run_text("w:t", new_text))
        rev_id += 1

    return rev_id


REVISIONS: dict[int, str] = {
    0: "中文题目：PhyFSME：面向非平稳多元时序预测的物理一致性多尺度分数阶谱混合专家框架",
    1: "英文题目：PhyFSME: A Physics-Consistent Multi-Scale Fractional Spectral Mixture-of-Experts Framework for Non-stationary Multivariate Time-Series Forecasting",
    2: "摘要",
    3: (
        "多元时间序列预测（Multivariate Time-Series Forecasting, MTSF）广泛服务于交通调度、能源管理、气象预警与工业控制等关键场景。"
        "真实世界序列通常同时包含长期趋势、局部突变、跨变量耦合以及频率随时间变化的非平稳动态，使得依赖固定时间窗口或固定傅里叶基底的模型难以获得稳定而紧凑的表示。"
        "为此，本文提出 PhyFSME，一种面向非平稳多元序列的双流预测框架。该框架在频域分支中引入可学习分数阶傅里叶变换（Fractional Fourier Transform, FRFT），"
        "通过连续可调的时频旋转将输入映射到更适合当前动态的分数阶谱空间；同时设计多尺度谱专家与尺度感知路由机制，自适应融合不同感受野下的频谱表示。"
        "为增强复数域优化的稳定性，本文进一步提出复数域物理一致性正则化，通过约束逆变换后虚部能量比例、局部波动和整体幅值，抑制无意义的虚部放大，并保留有助于相位结构建模的有效信息。"
        "此外，模型以变量交互分支显式建模跨变量依赖，并通过动态门控融合实现频域非平稳动态与时域变量关系的互补建模。"
        "在多个公开基准数据集上的实验、消融与鲁棒性分析表明，PhyFSME 能在预测精度、结构稳定性和噪声鲁棒性之间取得更好的平衡。"
    ),
    7: "核心关键词（Keywords）",
    8: "多元时间序列预测；分数阶傅里叶变换；混合专家；复数域稳定性正则化；非平稳信号建模",
    11: (
        "图 1. 非平稳时间序列中的多尺度结构与分数阶谱表示动机。"
        "(a) 真实序列往往同时包含低频宏观趋势与局部瞬态扰动：前者需要较大的时间感受野，后者需要更高的局部时间分辨率。"
        "(b) 标准傅里叶变换（α=1.0）使用固定正交基底，当信号频率随时间变化时容易产生谱能量弥散与泄漏，削弱局部非平稳模式的可分辨性。"
        "(c) 通过学习合适的分数阶角度（如 α=0.3），FRFT 可在连续时频平面中重新选择投影方向，使原本分散的能量重新聚焦，从而为后续预测提供更紧凑的结构化表示。"
    ),
    14: (
        "图 2. PhyFSME 的复数域一致性与结构鲁棒性验证。"
        "(a) 逆分数阶变换后虚部响应的概率密度分布。加入复数域稳定性正则化后，虚部能量更集中于零附近，说明模型能够更好地回到实值观测流形。"
        "(b) 虚部响应随时间步的变化。无约束模型更容易出现局部振荡，而引入正则化后虚部波动被显著抑制。"
        "(c) 噪声扰动下的预测误差变化。PhyFSME 在不同噪声强度下表现出更平缓的误差增长趋势，表明分数阶谱建模与复数域一致性约束有助于提升非平稳场景下的结构稳定性。"
    ),
    17: "1 引言",
    18: (
        "多元时间序列预测的核心目标是利用历史多变量观测推断未来动态，其应用覆盖智能交通、能源负荷、气象监测、金融风险与工业过程控制等场景。"
        "与单变量预测相比，MTSF 不仅要求模型刻画单个变量的时间演化，还需要同时捕获变量之间的协同关系、长期趋势与局部瞬态变化。"
        "在真实系统中，这些动态往往呈现明显的非平稳性与多尺度耦合：一方面，低频趋势决定长期演化方向；另一方面，局部突变和短期扰动可能快速改变未来状态。"
        "因此，一个面向高精度预测的模型应同时具备跨变量依赖建模能力、非平稳时频表征能力以及对噪声扰动的结构鲁棒性。"
    ),
    19: (
        "近年来，深度学习显著推动了 MTSF 的发展。时域模型（如 RNN、TCN 与 Transformer）通过递归、卷积或注意力机制捕获序列依赖；"
        "频域模型则借助傅里叶变换或周期重构在全局频率空间中提取周期结构。尽管这些方法在标准基准上取得了良好效果，但它们大多依赖固定的时间窗口、固定的频率基底或固定的尺度划分。"
        "当序列中存在频率漂移、局部 chirp 结构或突发扰动时，固定表示往往难以在统一空间中同时刻画长期周期与局部非平稳动态，进而导致谱能量弥散、特征耦合不充分以及泛化性能下降。"
    ),
    20: (
        "我们认为，现有方法在建模复杂非平稳多元系统时主要面临三类瓶颈。"
        "第一，固定谱表示限制了模型对时变频率结构的适应能力；标准傅里叶基底本质上对应固定的时频投影方向，难以充分表达介于纯时域与纯频域之间的连续动态。"
        "第二，静态尺度建模难以兼顾长期趋势与局部突变；单一 patch 或预设窗口往往只能服务于某一类时间分辨率。"
        "第三，复数域谱表示虽然携带幅值与相位信息，但在端到端训练中若缺少约束，逆变换后的虚部响应可能被异常放大，从而破坏实值时间序列预测所需的表示一致性与优化稳定性。"
    ),
    22: (
        "图 1 与图 2 从动机层面对上述问题进行了可视化说明。图 1 表明，在含有瞬态 chirp 扰动的非平稳信号中，标准傅里叶变换可能产生明显的能量泄漏；"
        "而通过学习合适的分数阶角度，FRFT 能够将能量重新聚焦到更紧凑的表示区域。图 2 进一步展示了复数域建模中的稳定性问题：无约束模型可能在逆变换后产生较强的虚部振荡，"
        "而本文提出的稳定性正则化能够有效抑制虚部异常扩张，并在噪声扰动下保持更平稳的误差变化。这些观察共同说明，可学习时频旋转、多尺度路由与复数域一致性约束是处理非平稳 MTSF 的关键。"
    ),
    23: (
        "基于上述动机，本文提出物理一致性多尺度分数阶谱混合专家模型 PhyFSME。具体而言，PhyFSME 采用双流结构：频域分支通过多尺度可学习 FRFT 专家捕获非平稳谱动态，"
        "变量交互分支则在变量维度建模全局依赖关系。随后，模型通过动态门控模块自适应融合两类互补表示。与传统频域方法不同，PhyFSME 并不将序列固定投影到标准傅里叶空间，"
        "而是通过可学习分数阶参数为不同尺度专家寻找更合适的时频坐标；与简单多尺度拼接不同，Scale-MoE 根据输入内容动态分配尺度贡献；与无约束复数域建模不同，"
        "本文的稳定性正则化显式约束逆变换后虚部响应，使模型在保持相位结构表达能力的同时提升训练稳定性。"
    ),
    24: "本文的主要贡献如下：",
    25: "1）提出一种面向非平稳多元时间序列预测的分数阶谱建模框架，通过可学习 FRFT 在连续时频表示空间中自适应选择投影方向，从而缓解固定傅里叶基底带来的谱能量弥散问题。",
    26: "2）设计一种多尺度分数阶谱混合专家机制（Scale-MoE），使模型能够根据输入内容自适应融合不同 patch 尺度下的谱表示，同时捕获长期趋势、稳定周期与局部瞬态变化。",
    27: "3）提出复数域物理一致性正则化，从虚部能量比例、局部平滑性与整体幅值三个角度约束逆变换后的复数表示，提升端到端训练的稳定性和噪声鲁棒性。",
    28: "4）在多个公开 MTSF 基准上进行系统实验，包括主结果比较、模块消融、分数阶可解释性、噪声鲁棒性与复杂度分析，为所提方法的有效性提供多维证据。",
    30: "2 相关工作",
    31: (
        "时域依赖建模是多元时间序列预测的基础方向。早期方法主要依赖 RNN、LSTM、GRU 与 TCN 捕获局部或中短期依赖；"
        "随后，Transformer 及其变体通过自注意力机制扩大感受野，并通过稀疏注意力、patch 化或变量 token 化等策略降低长序列建模开销。"
        "这类方法在跨变量依赖与长期关系建模方面具有优势，但通常直接在原始时间域操作，对频率漂移、非平稳周期和局部谱变化的显式建模能力仍然有限。"
    ),
    32: (
        "频域表征方法通过傅里叶变换、周期检测或频域滤波来捕获全局周期结构。代表性工作通常利用 DFT 将序列映射到固定频率空间，并在频域执行选择、滤波或特征交互。"
        "然而，标准傅里叶基底对应固定的正交投影，当信号包含时变频率或局部 chirp 模式时，固定频域表示容易产生能量泄漏。"
        "因此，仅依赖标准傅里叶空间可能不足以刻画复杂非平稳序列中的连续时频演化。"
    ),
    33: (
        "时频联合学习试图在时间分辨率与频率分辨率之间取得平衡，例如 STFT、小波变换或多尺度卷积等。"
        "这些方法为非平稳建模提供了更灵活的分析工具，但其窗口、基函数或尺度配置往往仍需预设，难以在训练过程中针对不同变量和不同样本自适应调整。"
        "FRFT 作为傅里叶变换的连续推广，可通过分数阶参数在时频平面中执行可调旋转，为学习数据驱动的时频坐标提供了自然机制。"
    ),
    34: (
        "物理一致性与鲁棒学习关注如何将结构先验引入深度模型。与基于显式微分方程的物理信息神经网络不同，本文关注的是复数域谱表示与实值时间序列之间的一致性："
        "模型可以在中间层使用复数谱表示以保留相位信息，但最终预测对象仍位于实值观测空间。若缺少约束，虚部响应可能在优化过程中被无意义放大，影响表示稳定性。"
        "因此，本文从虚部能量与平滑性角度构造稳定性正则化，使分数阶谱表示更符合实值预测任务的结构要求。"
    ),
    35: (
        "与上述研究相比，PhyFSME 的区别在于同时引入可学习分数阶时频旋转、多尺度谱专家路由和复数域一致性正则化。"
        "三者分别对应“在哪个时频坐标中观察序列”“以何种尺度刻画动态”以及“如何稳定复数域优化”三个关键问题，从而形成面向非平稳 MTSF 的统一建模框架。"
    ),
    36: "3 预备知识与理论基础（Preliminaries and Theoretical Foundations）",
    37: "3.1 连续分数阶傅里叶变换（Continuous Fractional Fourier Transform）",
    44: "3.2 时频相空间的旋转算子（Rotation in Time-Frequency Phase Space）",
    49: "3.3 基于 Chirp 调制的快速可微离散算子（Learnable Discrete FRFT via Chirp Modulation）",
    56: "在此范式下，分数阶次 α 被松弛为网络中可动态优化的连续参数。相比固定谱基底，模型能够以数据驱动方式学习更适合当前序列动态的时频投影方向，为非平稳结构提供更紧凑的表示。",
    57: "3.4 谱能量聚焦与复数域稳定表示（Spectral Energy Focusing and Stable Complex Representation）",
    58: "FRFT 为非平稳时间模式提供了比固定傅里叶基底更灵活的谱表示。对于 chirp-like 或频率随时间变化的成分，合适的分数阶次能够使谱响应更加集中，从而缓解能量弥散并提高表示紧凑性。",
    59: "与此同时，可学习的复数域变换也会带来优化稳定性风险。若虚部响应过大或随时间剧烈振荡，模型可能在中间表示中偏离实值时间序列对应的物理观测流形。",
    60: "为缓解该问题，本文引入复数域稳定性正则化目标：",
    62: "其中，",
    64: "第一项用于约束虚部能量在整体复数表示中的相对比例，",
    66: "第二项用于鼓励虚部响应沿时间维保持局部平滑，",
    68: "第三项用于限制虚部整体幅值，避免无意义扩张。",
    69: "这些约束共同稳定复数域优化过程，同时保留虚部中与相位和局部时频耦合相关的有效结构信息。",
    72: "4 方法",
    74: "4.1 问题定义",
    84: "图 3. PhyFSME 总体架构。模型由多尺度分数阶谱专家分支、变量交互分支和动态门控融合模块组成；前者刻画非平稳谱动态，后者建模跨变量依赖，最终通过门控融合得到预测表示。",
    85: "4.2 总体架构",
    86: (
        "本文提出双流协同框架 PhyFSME，用于同时建模多元时间序列中的跨变量依赖结构与非平稳谱动态。"
        "如图 3 所示，模型由分数阶谱流（Fractional Spectral Stream）和变量交互流（Variable Interaction Stream）组成，并通过动态门控机制进行统一融合。"
    ),
    87: (
        "该设计的核心动机在于：多元时间序列通常同时包含两类归纳偏置不同的结构——变量之间的全局依赖关系，以及信号在时频空间中的非平稳演化。"
        "单一建模范式往往难以同时刻画这两类结构。因此，PhyFSME 采用显式解耦策略，在两个互补子空间中分别建模，并在高层表示中自适应融合。"
    ),
    88: (
        "具体而言，给定输入序列，模型首先进行实例归一化以缓解分布漂移。随后，归一化后的表示被并行送入两个分支："
        "分数阶谱流在多个 patch 尺度上使用可学习 FRFT 专家提取非平稳谱表示，并通过 Scale-MoE 聚合不同尺度贡献；"
        "变量交互流将每个变量的完整历史窗口视为变量级 token，并利用自注意力捕获跨变量依赖。"
    ),
    89: "在此基础上，动态门控融合模块根据输入内容自适应整合频域动态信息与变量依赖信息。融合后的表示通过线性预测头映射到未来时间窗口，并经反归一化得到最终预测结果。",
    90: "总体而言，PhyFSME 在分数阶频域与变量交互空间之间建立互补建模机制，在保持计算效率的同时增强了对复杂非平稳多元序列的表达能力。",
    93: "4.3 物理一致性分数阶谱专家（Physics-Consistent Fractional Spectral Experts）",
    95: "4.3.1 可学习分数阶傅里叶变换（Learnable Fractional Fourier Transform）",
    103: "该公式中同时出现 FFT 与 IFFT 是合理的：基于 Ozaktas 分解的快速 FRFT 将连续核中的交叉项转化为 chirp 调制与卷积结构，而卷积可通过“FFT—逐点乘法—IFFT”高效实现。因此，FFT 与 IFFT 并非重复操作，而是快速可微离散 FRFT 的核心计算路径。",
    110: "4.3.2 自适应谱解耦（Adaptive Spectral Decoupling）",
    130: "4.3.3 复数域通道混合（Complex-valued Channel Mixing）",
    164: "4.3.4 复数域稳定性正则化（Complex-domain Stability Regularization）",
    165: "经过逆分数阶傅里叶变换后，第 i 个专家得到的时域表示仍可能是复数形式。这是符合建模逻辑的：频域处理改变了复数谱的幅值和相位，逆变换后的实部用于恢复与预测实值序列结构，虚部则反映残余相位和局部时频耦合信息；本文通过稳定性正则化约束该虚部，使其不破坏实值预测流形。",
    175: "（1）虚部能量比例约束",
    188: "（2）局部平滑约束",
    195: "（3）虚部幅值稀疏约束",
    199: "（4）整体复数域稳定性目标",
    208: "4.4 尺度混合专家机制（Scale Mixture-of-Experts）",
    213: "4.4.1 多尺度异构谱专家（Multi-scale Heterogeneous Spectral Experts）",
    232: "4.4.2 动态门控路由（Dynamic Gated Routing）",
    245: "4.4.3 自适应尺度特征聚合（Adaptive Scale Feature Aggregation）",
    260: "4.5 变量交互分支（Variable Interaction Branch）",
    282: "4.6 动态门控特征融合（Dynamic Gated Feature Fusion）",
    300: "4.7 联合优化目标（Joint Optimization Objective）",
    308: "对应地，第 i 个专家的复数域稳定性约束定义为：",
    311: "第一项用于约束虚部能量占比；",
    312: "第二项用于抑制虚部局部剧烈波动；",
    313: "第三项用于限制整体虚部幅值规模；",
    314: "β1、β2 与 β3 为对应的平衡系数。",
    321: "5 实验",
    323: "5.1 实验设置",
    324: (
        "数据集。实验应覆盖典型 MTSF 场景，包括 ETT、Electricity、Traffic、Weather、Solar、Exchange 以及交通传感器类 PEMS 数据集。"
        "对于每个数据集，建议统一采用公开协议划分训练、验证与测试集，并在多个预测长度（如 96、192、336、720；PEMS 可使用 12、24、48、96）上报告结果，以保证与主流工作可比。"
    ),
    325: "评价指标。本文采用 MSE 与 MAE 作为主要指标，并报告平均性能、不同预测长度下的结果以及必要的标准差。所有模型应在相同输入长度、训练轮次、早停策略和随机种子设置下比较。",
    327: "5.2 与主流方法的比较",
    328: (
        "主实验应将 PhyFSME 与经典线性模型、Transformer 系列、patch-based 模型、频域模型以及多尺度模型进行系统比较。"
        "在撰写结果分析时，应避免只报告平均提升，而应分别讨论不同数据类型和预测长度下的优势来源：例如在强周期数据上分数阶谱专家的能量聚焦作用，在高维变量数据上变量交互分支的贡献，以及在长预测长度下多尺度路由的稳定性。"
    ),
    330: "5.3 消融实验",
    331: (
        "消融实验用于验证 Scale-MoE、可学习 FRFT、复数域稳定性正则化、变量交互分支和动态融合模块的独立贡献。"
        "建议至少包含以下变体：固定 FFT 替代 FRFT、固定 α 替代可学习 α、单尺度专家替代多尺度专家、平均融合替代 Scale-MoE、去除复数域稳定性正则化、去除变量交互分支以及去除动态门控融合。"
        "通过这些对比可以证明性能提升并非来自参数量堆叠，而来自所提出建模机制的结构优势。"
    ),
    334: "5.4 可解释性分析：分数阶参数与尺度路由",
    335: (
        "为解释 PhyFSME 的内部决策机制，本文可视化不同数据集、变量和尺度专家学习到的分数阶参数 α，并分析其与谱能量聚焦程度之间的关系。"
        "同时，展示 Scale-MoE 对不同变量分配的尺度权重，以说明模型如何在长期趋势、稳定周期与局部瞬态之间进行自适应选择。"
    ),
    338: "5.5 噪声鲁棒性实验",
    339: (
        "为评估模型在非平稳扰动下的稳定性，建议在测试输入中加入不同强度的高斯噪声或局部脉冲扰动，并比较各模型预测误差随噪声强度变化的斜率。"
        "若 PhyFSME 的误差增长更平缓，则可进一步支持分数阶谱建模与复数域稳定性正则化能够提升结构鲁棒性的结论。"
    ),
    341: "5.6 模型复杂度分析",
    342: "复杂度分析应同时报告参数量、训练时间、推理时间和显存消耗。理论上，FRFT 专家主要由 FFT/IFFT 及轻量级通道混合构成，复杂度与序列长度呈 O(L log L) 关系；变量交互分支的注意力开销与变量数相关，因此在高维数据集上需要重点比较效率与精度之间的权衡。",
    343: "6 结论",
    344: (
        "本文提出 PhyFSME，一种面向非平稳多元时间序列预测的物理一致性多尺度分数阶谱混合专家框架。"
        "通过可学习 FRFT，模型能够在连续时频空间中自适应寻找更合适的谱投影方向；通过 Scale-MoE，模型能够根据输入内容融合不同时间尺度的谱专家；通过复数域稳定性正则化，模型能够抑制无意义虚部响应并提升实值预测的一致性。"
        "未来工作可进一步探索更细粒度的变量级分数阶参数、面向超长序列的稀疏谱专家，以及与外生事件或图结构先验的联合建模。"
    ),
    352: "附录 A：连续到离散的核函数解耦推导",
    353: "标题：分数阶傅里叶变换快速计算的推导（Ozaktas 分解）",
    358: "利用基本三角恒等式对交叉项进行重写，并将其代入相位函数展开：",
    360: "合并含有 t 和 u 的项后，可将交叉耦合项整理为完全平方形式：",
    371: "通过上述离散化过程，可以得到正文第 3.3 节采用的可微张量计算图。该推导说明，代码中的 FFT 与 IFFT 来自快速 FRFT 的卷积实现，并不与 FRFT 本身矛盾；同时，复数乘法、FFT 与 IFFT 均支持反向传播，因此可学习分数阶参数 α 能够在端到端训练中持续更新。",
}


DELETE_ONLY = list(range(377, 416)) + list(range(431, 455))


def enable_track_revisions(settings_xml: Path) -> None:
    tree = etree.parse(str(settings_xml))
    root = tree.getroot()
    if root.find(qn("w:trackRevisions")) is None:
        root.insert(0, etree.Element(qn("w:trackRevisions")))
    tree.write(str(settings_xml), xml_declaration=True, encoding="UTF-8", standalone="yes")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(SOURCE) as zin:
            zin.extractall(tmpdir)

        doc_xml = tmpdir / "word" / "document.xml"
        settings_xml = tmpdir / "word" / "settings.xml"
        tree = etree.parse(str(doc_xml))
        root = tree.getroot()
        body = root.find(qn("w:body"))
        paragraphs = body.findall(qn("w:p"))

        rev_id = next_revision_id(root)

        edits = dict(REVISIONS)
        for idx in DELETE_ONLY:
            edits.setdefault(idx, "")

        for idx, new_text in sorted(edits.items()):
            if idx >= len(paragraphs):
                continue
            rev_id = replace_paragraph_with_revision(paragraphs[idx], new_text, rev_id)

        tree.write(str(doc_xml), xml_declaration=True, encoding="UTF-8", standalone="yes")
        enable_track_revisions(settings_xml)

        if OUTPUT.exists():
            OUTPUT.unlink()
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for file in tmpdir.rglob("*"):
                if file.is_file():
                    zout.write(file, file.relative_to(tmpdir).as_posix())

    print(OUTPUT)


if __name__ == "__main__":
    main()
