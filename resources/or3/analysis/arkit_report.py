"""Evidence-bound Chinese report, manuscript passage, and compact replication bundle."""
from pathlib import Path
import csv,json,zipfile
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent;N=ROOT/'new_scene_validation';S=N/'summary';OUT=ROOT.parents[1]/'outputs'
plan=json.loads((N/'evaluation_plan_frozen.json').read_text(encoding='utf-8'))
validation=json.loads((S/'validation.json').read_text(encoding='utf-8'))
p=pd.read_csv(S/'paired_summary.csv');a=pd.read_csv(S/'aggregate_metrics.csv')
v=pd.read_csv(S/'venue_metrics.csv');seeds=pd.read_csv(S/'paired_seeds.csv')
regions=pd.read_csv(S/'venue_regions.csv');agg=pd.read_csv(S/'aggregate_regions.csv')
cohort=pd.read_csv(S/'cohort_venues.csv')
def row(df,**kw):
 for k,x in kw.items():df=df[df[k]==x]
 assert len(df)==1,kw
 return df.iloc[0]
def pct(x,digits=2):return f'{x:+.{digits}f}%'
primary=row(p,family='unseen8',aggregation='venue',control='image_mean')
parent=row(p,family='unseen8',aggregation='venue',control='parent')
image=row(p,family='unseen8',aggregation='image',control='image_mean')
metric_table=['| 家族 | Parent（m） | Spatial（m） | Image Mean（m） | Spatial 相对 Parent | Spatial 相对 Image Mean |',
 '|---|---:|---:|---:|---:|---:|']
for family in ['unseen8','uniform','seen2']:
 vals=[row(a,family=family,aggregation='venue',method=x).rmse for x in ['parent','spatial','image_mean']]
 changes=[row(p,family=family,aggregation='venue',control=x).change_percent for x in ['parent','image_mean']]
 metric_table.append(f'| {family} | '+' | '.join(f'{x:.6f}' for x in vals)+' | '+' | '.join(pct(x) for x in changes)+' |')
seed_table=['| 训练种子 | Spatial（m） | Image Mean（m） | 配对差值（m） | 相对 Image Mean | 相对 Parent | 相对 Image Mean 受益场所 |','|---|---:|---:|---:|---:|---:|---:|']
ps=seeds[(seeds.family=='unseen8')&(seeds.control=='image_mean')]
for r in ps.itertuples():
 parent_seed=row(seeds,family='unseen8',seed=r.seed,control='parent').change_percent
 seed_table.append(f'| {r.seed} | {r.spatial_rmse:.6f} | {r.control_rmse:.6f} | {r.delta_m:+.6f} | {pct(r.change_percent)} | {pct(parent_seed)} | {r.venue_wins}/20 |')
wide=v[v.family=='unseen8'].pivot(index='scene',columns='method',values='rmse')
venue_table=['| 场所 visit_id | Parent（m） | Spatial（m） | Image Mean（m） | Spatial 相对 Image Mean |','|---|---:|---:|---:|---:|']
for visit,r in wide.sort_index().iterrows():venue_table.append(f'| {visit.removeprefix("visit_")} | {r.parent:.6f} | {r.spatial:.6f} | {r.image_mean:.6f} | {pct(100*(r.spatial/r.image_mean-1))} |')
reg=regions[regions.control=='image_mean'];rwide=reg.pivot(index='scene',columns='region',values='delta_mse')
far_wins=int((rwide.far<0).sum());near_wins=int((rwide.near_unobserved<0).sum());middle_wins=int((rwide.middle<0).sum())
pattern=int(((rwide.far<0)&(rwide.near_unobserved>0)&(rwide.middle>0)).sum())
region_table=['| 区域 | 对场所等权 Image Mean 全图 MSE 的归一化差异贡献 | 该区域 MSE 受益场所 |','|---|---:|---:|']
for name,label in [('input','输入观测点'),('near_unobserved','近区未观测点'),('middle','中区'),('far','远区'),('all','全图合计')]:
 r=row(agg,control='image_mean',region=name);wins=int((reg[reg.region==name].delta_mse<0).sum())
 region_table.append(f'| {label} | {pct(r.contribution_percent,4)} | {wins}/20 |')
signs=[int((ps.delta_m<0).sum()),int((ps.delta_m>0).sum())]
sd=float(ps.delta_m.std(ddof=1))
direction='降低' if primary.change_percent<0 else ('增加' if primary.change_percent>0 else '持平')
parent_direction='降低' if parent.change_percent<0 else '增加'
if primary.change_percent<0:
 conclusion=f'在本轮新场所主终点上，Spatial 相对 Image Mean 的平均 RMSE 降低 {abs(primary.change_percent):.2f}%；受益覆盖为 {int(primary.wins)}/20 个场所。平均变化、训练种子方向与受益覆盖应并列解释。'
else:
 conclusion=f'在本轮新场所主终点上，Spatial 未复现相对 Image Mean 的平均收益，RMSE 增加 {primary.change_percent:.2f}%；受益覆盖为 {int(primary.wins)}/20 个场所。论文应据此报告空间门控增量的条件依赖。'
if parent.change_percent>0:
 conclusion+=f'与此同时，Spatial 相对 Parent 的平均 RMSE 增加 {parent.change_percent:.2f}%，仅 {int(parent.wins)}/20 个场所受益；相对常值门控的设计增量不等于整体适配优于父模型。'
protocol=(OUT/'新场景验证协议与结果.md').read_text(encoding='utf-8')
protocol=protocol.replace('状态：协议已冻结，模型评价运行中；结果待完整核验后填写。','状态：全量评价与核验已完成。')
protocol=protocol.split('**结果：等待全量评分与核验完成。**')[0].split('**主结果。**')[0]
report=protocol+f'''**主结果。** {conclusion}

下表为场所等权的平均逐图 RMSE。方法分支对三个训练种子的指标求平均，Parent 只有一个冻结状态；未对模型预测做集成。相对变化为 Spatial / 对照 − 1，负值表示 Spatial 更好。

{chr(10).join(metric_table)}

主要 unseen8 的绝对配对差值为 {primary.delta_m:+.6f} m。图像等权结果为 {pct(image.change_percent)}，图像级受益覆盖 {int(image.wins)}/{int(image.units)}；它与场所等权结果对应不同的采集分配权重。相对 Parent，Spatial 在 {int(parent.wins)}/20 个场所改善。uniform 与 seen2 属于新目标域的补充布局评价，不能代替原 NYUv2 的源域保持评价。

**训练种子重复。**

{chr(10).join(seed_table)}

配对差值均值 ± 样本标准差为 {primary.delta_m:+.6f} ± {sd:.6f} m，n=3。三个种子中 {signs[0]} 个支持 Spatial、{signs[1]} 个支持 Image Mean。该标准差衡量训练重复间变化，不是新场所总体效应的置信区间。

相对 Parent 的训练种子方向不同：种子 43 改善，种子 42、44 退化。因此，平均高于 Parent 的结果应与这组种子差异一起报告。

**全部场所的配对结果。**

{chr(10).join(venue_table)}

以上场所均按原清单保留，没有因为误差大或比较方向不利而移除。每场所帧数为 3—15，总图像数 247；场所等权主终点避免长视频和多帧场所获得额外权重。

**DIODE 区域观察能否迁移。** 使用与先前 DIODE 诊断相同的归一化最近观测距离分区，输入观测点单列。区域贡献用各图像的平方误差和差值除以全图有效像素数，先在场所内平均，再对场所等权；百分比以 Image Mean 的场所等权全图平均逐图 MSE 为分母。

{chr(10).join(region_table)}

远区有 {far_wins}/20 个场所受益，近区未观测点有 {near_wins}/20 个受益，中区有 {middle_wins}/20 个受益。满足“远区改善，同时近区未观测点和中区均退化”的场所有 {pattern}/20。该符号模式复核的是预先指定的区域观察，不能替代主要 RMSE 对照，也不能单独证明门控位置的因果作用。完整各场所、各训练种子、交叉项及修正平方项见支撑表。

新队列的场所等权平均有效深度为 {cohort.mean_depth_m.mean():.3f} m，各场所均值范围为 {cohort.mean_depth_m.min():.3f}—{cohort.mean_depth_m.max():.3f} m。较浅的新队列仍出现广泛远区收益，因此不能把 DIODE 场景 00019 的较大平均深度直接当作跨队列收益选择条件。这里的远区指远离输入观测点，与物理深度大小是不同变量；域、几何和父模型误差结构的影响仍需分辨。

**对投稿内容的影响。** 本轮增加了一个新场所队列，补上了前稿只有三个 DIODE 物理场景、第二父模型仍复用同一队列的关键证据缺口。{conclusion} 新队列只运行现有 CompletionFormer 的三个分支条件；OMNI-DC 在新队列上的迁移尚未执行。正文可以把本轮作为独立小节，连接原 DIODE 的平均收益、采集单元覆盖和区域机制分析，按实际结果更新摘要与结论。

两组队列构成互补证据：DIODE 上相对 Parent 的适配恢复覆盖全部扫描，而相对 Image Mean 的增量集中于一个扫描；ARKitScenes 上相对 Image Mean 的增量覆盖绝大多数场所，但种子平均的整体适配仍弱于 Parent。稿件原有“空间门控增量覆盖有限”的概括应限定到 DIODE，整体论点应改为三个评价对象随队列变化而分离，不能将当前新结果解释为普遍覆盖不足。

**核验。** 已检查 {validation['score_rows']:,} 条完整评分和 {validation['region_rows']:,} 条区域对照记录；所有逐图文件的哈希、方法×布局×采样组合、配对采样坐标、固定 500 点和评分像素数一致。{validation['whole_image_mse_checks']:,} 组区域全图 MSE 与对应 RMSE 平方吻合，分区可加性及 2ec+c² 恒等式通过，最大代数绝对残差 {validation['max_algebra_residual']:.3g}。另一条数据核查从原始 PNG 的像素索引重新验证全部 247 张深度的方向、采样位置及单位，核对全部 13,585 组 500 点支持。评价启动时间晚于冻结计划时间。原模型检查点及源代码身份在推理入口再次核对。

**复核材料。** 同目录 `ARKitScenes新场所验证材料.zip` 包含冻结计划、清单与采样表、下载校验记录、完整逐条件评分与区域记录、汇总表及本轮分析代码。原始数据压缩包与转换图像保留在本任务工作目录，未重复装入交付压缩包。图表 `ARKitScenes逐场所验证` 和 `新旧场景区域误差对照` 导出 PNG、PDF 与 SVG。
'''
(OUT/'新场景验证协议与结果.md').write_text(report,encoding='utf-8')
manuscript=f'''**新场所验证：可合并到正文的小节**

为检验设计增量及其空间收益模式在新场所的可迁移性，我们在 ARKitScenes 深度上采样子集上开展冻结模型验证。实验包含官方 Validation 分区的 20 个场所、247 张配对图像；按 visit_id 分组，每个场所选一个视频，并在视频内按时间最多等距抽取 15 帧。预处理检查使用另一个场所。所有模型、采样规则及主终点在评价场所预测前冻结，此时已知前述 DIODE 结果。源域训练和检查点保持不变。

每张图像采用参考深度合成的 500 点稀疏输入、相同 11 种布局及五次配对采样，主评价仍为 unseen8 下的场所等权平均逐图 RMSE。RGB 与参考深度按元数据统一方向，缩放后裁切至 304×228；有效参考深度范围为 0.1—10 m。该实验检验新场所上的设计增量，输入仍由参考深度采样生成。

在场所等权主终点上，Parent、Spatial 与 Image Mean 的平均逐图 RMSE 分别为 {row(a,family='unseen8',aggregation='venue',method='parent').rmse:.6f}、{primary.spatial_rmse:.6f} 和 {primary.control_rmse:.6f} m。Spatial 相对 Parent {parent_direction} {abs(parent.change_percent):.2f}%，相对 Image Mean {direction} {abs(primary.change_percent):.2f}%。前一比较的受益场所为 {int(parent.wins)}/20，后一比较为 {int(primary.wins)}/20。这表明，相对常值门控的广泛设计收益可以与整体适配未优于父模型同时出现。

相对 Image Mean，图像等权变化为 {pct(image.change_percent)}，三个训练种子的配对 RMSE 差值为 {primary.delta_m:+.6f}±{sd:.6f} m（均值±样本标准差）。三种子在该对照下均改善。相对 Parent，种子 43 改善，种子 42、44 退化，因此总体均值的方向还需结合训练重复解释。

按与 DIODE 相同的最近观测距离分区，远区、近区未观测点和中区的 MSE 分别在 {far_wins}/20、{near_wins}/20 和 {middle_wins}/20 个场所改善。满足远区改善而近中区同时退化的场所有 {pattern}/20。区域差异说明了各条件下收益与代价的分布，空间门控在新场所上的表现应结合平均误差、受益覆盖及区域贡献共同解释。

表：新场所验证的场所等权平均逐图 RMSE（m）。负相对变化表示 Spatial 更好。

{chr(10).join(metric_table)}

图注（ARKitScenes逐场所验证）：unseen8 下 Spatial 相对 Parent 与 Image Mean 的逐场所平均逐图 RMSE 差值。场所内先对布局、采样和图像平均，适配分支再对三个训练种子的指标平均；负值表示 Spatial 更好。两个面板共用横轴尺度，全部 20 个场所均按冻结清单报告。

图注（新旧场景区域误差对照）：Spatial 相对 Image Mean 的 unseen8 区域 MSE 差异贡献，分别按 DIODE 的三个场景及 ARKitScenes 的 20 个场所等权。各区域以所在数据队列的 Image Mean 全图平均逐图 MSE 归一化，黑色菱形表示净变化。该图比较区域贡献模式，不用于两数据集绝对误差排名。

新增引用：ARKitScenes，Baruch 等，NeurIPS Datasets and Benchmarks 2021；官方论文 https://arxiv.org/abs/2111.08897 。

**原稿需同步的表述。**

摘要中，保留 DIODE 的主要结果及其集中收益观察，增加一句：在新增的 ARKitScenes 20 个场所、247 张图像上，Spatial 相对 Image Mean 的场所等权 RMSE {direction} {abs(primary.change_percent):.2f}%，受益场所为 {int(primary.wins)}/20；相对 Parent 的平均 RMSE 则{parent_direction} {abs(parent.change_percent):.2f}%。新增结果使用独立句，避免与 OMNI-DC 的同队列复核混淆。

引言贡献中，将证据链表述为：以相同配对观测区分整体适配恢复和空间门控设计增量，并结合 DIODE 分层诊断、同队列第二父模型复核及 ARKitScenes 新场所验证，考察平均收益与受益范围的关系。

局限性中，将“尚未增加独立新场景”更新为：新场所验证覆盖 ARKitScenes 的 20 个符合方向条件的场所，使用冻结 CompletionFormer 和参考深度合成输入；该结果未覆盖设备原生稀疏输入，也未在新场所上复核 OMNI-DC。场所内最多 15 帧的抽样及官方公开分区共同定义了评价范围。

结论中可加入：{conclusion} 跨队列结果应与各自的物理采集单位和配对对照共同解释。
'''
(OUT/'新场景验证正文补充.md').write_text(manuscript,encoding='utf-8')
archive=OUT/'ARKitScenes新场所验证材料.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for name in ['selection.json','archive_inventory.json','evaluation_manifest.json','evaluation_supports.csv','evaluation_plan_frozen.json','evaluation_started.json','inference_complete.json','preflight.json','preprocessing_checks.json','data_verification.json','download_all.json']:
  z.write(N/name,'protocol/'+name)
 for pth in (N/'archives').glob('*.json'):z.write(pth,'archive_receipts/'+pth.name)
 for pth in S.iterdir():z.write(pth,'summary/'+pth.name)
 for pth in (N/'per_image').iterdir():
  if pth.suffix in ['.csv','.json']:z.write(pth,'per_image/'+pth.name)
 for name in ['arkit_acquire.py','arkit_prepare.py','arkit_evaluate.py','arkit_analyze.py','arkit_verify_data.py','arkit_plot.py','arkit_report.py','diode_mechanism.py']:
  z.write(ROOT/name,'analysis/'+name)
 for name in ['splits.csv','metadata.csv','receipts.json','README.md','DATA.md','LICENSE']:z.write(N/'source'/name,'source/'+name)
 z.writestr('说明.txt','ARKitScenes 新场所验证：本轮为 CompletionFormer 冻结模型；并非 OMNI-DC 新场所结果。\n主终点为 unseen8 场所等权平均逐图 RMSE，区域表是可加 MSE 分解。\n输入来自参考深度合成的 500 点；设备稀疏输入不在本轮结果中。\n保留本机来源路径以供本地复算，脚本不是跨机器安装包。\n原始官方数据及模型权重未重复打包，下载校验记录与来源地址随附。\n')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print('Report, manuscript supplement and evidence ZIP saved.')
print(conclusion)
