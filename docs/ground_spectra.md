# SpectraAlign-FM 地面光谱曲线

这份数据对应 SpectraAlign-FM 论文 §4.3 的地面光谱参考库，包含最终联合预训练所用的 **30,427 条原型训练条目**。四个传感器以这一批曲线为输入，按各自波段响应重采样、筛选和聚类。来源总目录规模为 233,097 条，本文以下统计以实际原型训练池为准。

## 下载

提取码 `phmx`。 [百度网盘](https://pan.baidu.com/s/1O0Hqcc5m6LFRh_MDBokMgw?pwd=phmx)

## 数据规模

| 材料类别 | 原型训练条目 | 加权前记录 |
| --- | ---: | ---: |
| 植被 | 10,800 | 10,800 |
| 土壤 | 8,623 | 8,623 |
| 水体 | 6,000 | 6,000 |
| 岩矿 | 2,480 | 2,480 |
| 人工材料 | 2,524 | 631 |
| **合计** | **30,427** | **28,534** |

- 植被、土壤和岩矿沿用服务器最终选中的曲线。人工材料的 631 条曲线各保留原条目并增加 3 个同值副本，以提高它们在聚类中的权重。索引通过 `original_uid` 将 1,893 个副本指回原条目。
- 水体条目包含 Bi 与 Hieronymi 的模拟训练光谱，以及波罗的海和 GLORIA 的实测遥感反射率。原型训练归档将每条水体曲线在原来源波长上限之后延伸到 2,500 nm，延伸段光谱值为零。`zero_fill_from_nm` 记录该上限。归档把延伸段的 `quality_mask` 设为 `0`，与实际构建器采用的有效波段约定一致。延伸值体现预处理假设。 [OWT 模拟方法](https://doi.org/10.1002/lno.12606) · [波罗的海船测数据](https://zenodo.org/records/5572537) · [GLORIA 数据](https://doi.pangaea.de/10.1594/PANGAEA.948492)

| 原型库 | 筛选后用于构建的条目 | 传感器波段 | 原型数 |
| --- | ---: | ---: | ---: |
| EO-1 Hyperion | 25,375 | 175 | 256 |
| GF-5 AHSI | 27,782 | 150 | 256 |
| DZ1 VNIR | 28,070 | 16 | 256 |
| DZ1 SWIR | 17,487 | 10 | 256 |

上述构建统计来自最终训练配置指向的四个原型库。DZ1 SWIR 构建的零信号筛选排除了 6,000 条水体延伸曲线。

## 原始光谱库和数据入口

条目数来自最终训练清单。人工材料按实际聚类权重计数。表中原始数据入口指向发布者、托管库或原作者项目。GOSPEL 三个中国子集共用国家数据中心公布的母库入口。

| 类别 | 原始光谱库或数据集 | 实验条目 | 数据入口 |
| --- | --- | ---: | --- |
| 植被 | EcoSIS（Ecological Spectral Information System）叶片与冠层光谱 | 9,727 | [EcoSIS](https://ecosis.org/) · [Cedar Creek 子库示例](https://doi.org/10.21232/hCtqmGdo) |
| 植被 | Tross 等的内布拉斯加玉米和高粱叶片高光谱反射率 | 576 | [Figshare v4，含实际使用的三个 CSV](https://doi.org/10.6084/m9.figshare.24808491.v4) |
| 植被 | 2001–2018 年中国植被反射率数据集 | 425 | [GOSPEL 典型地表类型反射率数据集](https://www.geodata.cn/data/datadetails.html?dataguid=137928179833049&docId=15040) |
| 植被 | Zhang 的红树林植物叶片反射光谱 | 72 | [Zenodo 数据记录，文件访问受限](https://doi.org/10.5281/zenodo.8312725) |
| 土壤 | LUCAS 2009 TOPSOIL 光谱 | 3,160 | [欧盟 JRC/ESDAC 数据申请和许可](https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data) |
| 土壤 | Open Soil Spectral Library（OSSL）v1.2 Vis–NIR 中的 KSSL 条目 | 3,123 | [OSSL v1.2 实验所用压缩 CSV](https://storage.googleapis.com/soilspec4gg-public/ossl_visnir_L0_v1.2.csv.gz) · [OSSL 下载文档](https://docs.soilspectroscopy.org/db-access.html) · [USDA KSSL 数据库入口](https://www.nrcs.usda.gov/conservation-basics/soil/soil-apps-tools) |
| 土壤 | 巴西南部土壤光谱（SOC Harmonization 项目的 `bes.csv`） | 979 | [Horst-Heinen 项目的源文件](https://github.com/taciaraz/soc-harmonization/blob/master/data/bes.csv) · [上游 BSSL 数据发布](https://zenodo.org/records/8361419) |
| 土壤 | 匈牙利土壤退化观测系统 HSDOS Vis–NIR 光谱库 | 942 | [Zenodo v1.1，HSDOS_SSL_ver1.1.csv](https://zenodo.org/records/13955229) |
| 土壤 | 2008–2018 年中国土壤反射率数据集 | 419 | [GOSPEL 典型地表类型反射率数据集](https://www.geodata.cn/data/datadetails.html?dataguid=137928179833049&docId=15040) |
| 水体 | Bi 与 Hieronymi 的 Optical Water Type 高光谱模拟训练集（`owt_BH2024_training_data_hyper.nc`） | 2,655 | [Zenodo](https://zenodo.org/records/12803329) |
| 水体 | 2016 年波罗的海船载高光谱遥感反射率（`rrs-bal-filtered.csv`） | 2,536 | [Zenodo](https://zenodo.org/records/5572537) |
| 水体 | GLORIA 内陆和沿海水域遥感反射率（`GLORIA_Rrs.csv`） | 809 | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.948492) |
| 岩矿 | 1981–2016 年中国岩矿反射率数据集 | 1,089 | [GOSPEL 典型地表类型反射率数据集](https://www.geodata.cn/data/datadetails.html?dataguid=137928179833049&docId=15040) |
| 岩矿 | Janice Bishop 岩矿及火星类比材料光谱（RELAB 测量，`BKR1JB*.TXT`） | 443 | [NASA PDS 的 RELAB 光谱档案](https://pds-geosciences.wustl.edu/speclib/urn-nasa-pds-relab/) |
| 岩矿 | USGS Spectral Library Version 7（`splib07a`） | 392 | [USGS 数据发布](https://doi.org/10.5066/F7RR1WDJ) |
| 岩矿 | GREENPEG 欧洲伟晶岩、矿物与围岩光谱库 | 300 | [Zenodo](https://zenodo.org/records/6518319) |
| 岩矿 | Fregeneda–Almendra 锂矿物与伟晶岩光谱库 | 152 | [Zenodo](https://zenodo.org/records/4575375) |
| 岩矿 | 北爱尔兰 Mourne Mountains 岩石可见光至短波红外光谱 | 76 | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.956842) |
| 岩矿 | NASA PDS Frankenspectra 矿物拼接光谱 | 21 | [NASA PDS 原始数据包](https://pds.nasa.gov/ds-view/pds/viewBundle.jsp?identifier=urn%3Anasa%3Apds%3Afrankenspectra_database&version=1.0) |
| 岩矿 | Hinrichs 与 Lucey 的矿物、陨石和月壤温度依赖光谱 | 7 | [Zenodo 原始光谱文件](https://zenodo.org/records/4980831) |
| 人工材料 | Salcido 与 Laefer 整编的 Existing Urban Hyperspectral Reference Data | 1,388 | [Zenodo，含上游光谱库来源与引用](https://zenodo.org/records/10963442) |
| 人工材料 | Garaba 等的干湿塑料、沥青和混凝土熟料反射率（350–1,000 nm 文件） | 604 | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.937105) |
| 人工材料 | LUMA/SLUM 不透水城市材料短波反射率 | 276 | [Zenodo](https://zenodo.org/records/4263842) · [雷丁大学发布页](https://micromet.reading.ac.uk/spectral-library/) |
| 人工材料 | Dileep 与 Deborah 的干燥和水下陶瓷校准砖高光谱数据 | 256 | [Zenodo，含 Data.zip 和 Readme.pdf](https://doi.org/10.5281/zenodo.8192707) |
| **合计** | **24 组来源** | **30,427** | |

城市整编数据的发布页附有上游来源清单。它包括 Karlsruhe Library of Urban Materials（KLUM）、ECOSTRESS、LUMA/SLUM 和其他城市材料库。可从发布页按原始库进行署名。 [整编数据及上游引用](https://zenodo.org/records/10963442) · [KLUM](https://github.com/rebeccailehag/KLUM_library)

## 文件格式和使用

解压后以同时包含 `native/` 和 `metadata/` 的目录作为数据根。

| 路径 | 内容 |
| --- | --- |
| `native/<材料类别>/<uid>.npz` | 原型构建输入的单条光谱，包含 `wavelengths`（nm）、`reflectance` 和 `quality_mask`。`quality_mask == 0` 为构建器识别的有效波段。水体的 `reflectance` 字段以原库的遥感反射率数据为来源，并保留原型训练使用的零值延伸。 |
| `native/<材料类别>/<uid>.json` | 逐条元数据，包括地物类、来源、原始文件名和增广信息。 |
| `metadata/ground_master_catalog.csv` | 可用 `utf-8-sig` 读取的逐条索引，含数据相对路径、来源、增广类型和对应 NPZ 的 SHA-256。 |
| `metadata/ecosis_collections.csv` | EcoSIS 的 127 个入选子库及其条目数、原文件名和已找到的原始入口。116 个子库已匹配原始题名，71 个已定位数据集 DOI。其余条目可按索引里的题名或原始文件名在 EcoSIS 查询。 |

光谱数组与服务器训练输入按字节一致。元数据整理为包内相对路径，保留复现与来源追溯字段。采样地点、日期和附加观测属性可按各原库资料查询。光谱数组维持训练输入的波长轴和质量码。

```python
import numpy as np

with np.load("native/water/water_0000001__waterzero2500.npz", allow_pickle=False) as spectrum:
    wavelengths_nm = spectrum["wavelengths"]
    reflectance = spectrum["reflectance"]
    valid_for_prototype = spectrum["quality_mask"] == 0
```

复现传感器原型时，将解压目录传给项目构建脚本的 `--ground-root`。 [SpectraAlign-FM 原型构建说明](../pretraining/README.md)

## 数据使用和引用

引用 SpectraAlign-FM 实验时，请同时按上表的原始数据页和 EcoSIS 子库索引署名。各数据源适用各自的许可和访问条件。

- **LUCAS 2009 · 3,160 条。** JRC 的官方许可限获授权者及其机构内按需人员访问，禁止转授权。公开分享这部分曲线前请核对实际取得的权限。读者也可从官方入口单独申请。 [JRC/ESDAC 许可和申请](https://esdac.jrc.ec.europa.eu/content/lucas-2009-topsoil-data)
- **波罗的海船测数据 · 2,536 条。** Zenodo 发布记录采用 CC BY-NC 4.0，使用时须署名并遵循非商业条款。 [Zenodo 数据与许可](https://zenodo.org/records/5572537)
- **红树林叶片光谱 · 72 条。** Zenodo 保留公开的书目信息，文件访问设为受限。可通过该记录联系提供方。 [Zenodo 数据记录](https://doi.org/10.5281/zenodo.8312725)

来源核对日期 2026-11-23。
