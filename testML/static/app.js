const { createApp, ref, computed, onMounted } = Vue;

window.__APP_BOOT_OK__ = true;
document.body.classList.remove('app-shell-pending');
document.body.classList.add('app-shell-ready');

createApp({
    setup() {
        const backendConnected = ref(true);
        const currentTab = ref(1);
        const tabs = ['1. Blueprint', '2. EDA Suite', '3. Preprocess', '4. Core Models', '5. Interpretation', '6. Optimization', '7. Figure Studio'];

        const fileData = ref(null);
        const selectedTarget = ref("");
        const problemType = ref("");

        const edaColX = ref("");
        const edaColY = ref("");
        const edaPlotType = ref("hist");
        const edaDpi = ref(300);
        const edaFormat = ref("png");
        const edaLoading = ref(false);
        const edaImage = ref(null);
        const edaPlotItems = ref([]);
        const auditLoading = ref(false);
        const auditResult = ref(null);
        const auditPanelOpen = ref(false);

        const prepOutlierMethod = ref("None");
        const prepScaler = ref("None");
        const prepAugment = ref("None");
        const prepFeatureMethod = ref("Random Forest");
        const prepTopK = ref(10);
        const featureImportances = ref([]);
        const prepOutlierPlot = ref("");
        const prepScalePlot = ref("");
        const prepAugmentPlot = ref("");
        const prepAugmentDiagnostics = ref({});
        const prepAugmentDiagnosticsLoading = ref(false);
        const prepFeaturePlot = ref("");
        const prepDpi = ref(300);
        const prepFormat = ref("png");

        const availableModels = ref([]);
        const selectedModels = ref([]);
        const trainTestSize = ref(20);
        const trainUseMeta = ref(false);
        const trainMetaAlgo = ref("PSO");
        const trainMetaCompareAll = ref(false);
        const trainMetaAlgorithms = ['PSO', 'GA', 'DE', 'GWO', 'WOA', 'SSA', 'HHO', 'SCA', 'SMA', 'AOA'];
        const trainEnsemble = ref("None");
        const trainLoading = ref(false);
        const trainResults = ref([]);
        const trainFailures = ref([]);
        const trainPlots = ref({});
        const globalPlots = ref({});
        const plotSelectedModel = ref("");
        const trainDpi = ref(300);
        const trainFormat = ref("png");
        const publicationLoading = ref(false);
        const publicationLeakage = ref(null);
        const publicationAblation = ref(null);
        const publicationExternal = ref(null);
        const publicationText = ref('');
        const publicationStoryboard = ref([]);
        const publicationAdvanced = ref(null);
        const publicationReadiness = ref(null);
        const publicationReviewer = ref(null);
        const publicationRobustness = ref(null);
        const publicationFigureBuilder = ref(null);
        const publicationExternalFileRef = ref(null);
        const publicationAblationTopK = ref(10);
        const publicationRobustnessRepeats = ref(8);
        const trainGlobalFilterEnabled = ref(false);
        const trainGlobalMinR2 = ref(0.9);
        const trainIntervalBins = ref(4);
        const trainIntervalEdges = ref('');
        const modelPlotViewMode = ref('all');
        const selectedModelPlotCategory = ref('all');
        const categoryOverviewWidth = ref(7.2);
        const categoryOverviewHeight = ref(9.7);
        const categoryOverviewColumns = ref(2);
        const categoryOverviewDpi = ref(300);
        const categoryOverviewFormat = ref('png');
        const modelPlotCategories = [
            { id: 'accuracy', label: '模型精度', icon: 'fa-bullseye', desc: '直接评价预测准确性、误差大小和排序能力。' },
            { id: 'generalization', label: '泛化稳定性', icon: 'fa-shield-halved', desc: '比较训练、测试、交叉验证和重采样表现是否稳定。' },
            { id: 'overfit', label: '过拟合检测', icon: 'fa-triangle-exclamation', desc: '识别训练表现过强但测试表现下降的风险。' },
            { id: 'residual', label: '残差结构', icon: 'fa-wave-square', desc: '分析残差分布、偏态、自相关、周期性和局部漂移。' },
            { id: 'calibration', label: '校准与不确定性', icon: 'fa-gauge-high', desc: '判断预测值、概率或区间是否可信。' },
            { id: 'risk', label: '误差风险', icon: 'fa-fire', desc: '定位高误差样本、尾部风险和最坏情形。' },
            { id: 'robustness', label: '适用域与鲁棒性', icon: 'fa-compass-drafting', desc: '评估外推、输入扰动、异常值和适用域可靠性。' },
            { id: 'feature', label: '特征与交互', icon: 'fa-diagram-project', desc: '解释误差和预测如何随特征或特征组合变化。' },
            { id: 'optimization', label: '优化过程', icon: 'fa-route', desc: '展示超参数和元启发式优化的搜索、收敛和组合对比。' },
            { id: 'classification', label: '分类专用', icon: 'fa-layer-group', desc: '分类任务的混淆矩阵、阈值、概率和类别表现。' },
            { id: 'summary', label: '综合总览', icon: 'fa-table-cells-large', desc: '雷达图、矩阵、排行榜和综合诊断面板。' },
            { id: 'other', label: '其他补充', icon: 'fa-ellipsis', desc: '暂未归入主类别的补充图。' },
        ];

        const shapLoading = ref(false);
        const shapPlots = ref({});
        const advancedXaiLoading = ref(false);
        const advancedXaiPlots = ref({});
        const shapSampleIdx = ref(0);
        const pdpFeature = ref("");
        const pdpFeature2 = ref("None");
        const pdpFeature3 = ref("None");
        const pdpLoading = ref(false);
        const pdpPlots = ref({});
        const permutationLoading = ref(false);
        const permutationPlots = ref({});
        const counterfactualSampleIdx = ref(0);
        const counterfactualDesiredValue = ref('');
        const counterfactualLoading = ref(false);
        const counterfactualResult = ref(null);
        const causalTreatment = ref("");
        const causalLoading = ref(false);
        const causalResult = ref(null);
        const interpDpi = ref(300);
        const interpFormat = ref("png");
        const optimizationSchema = ref(null);
        const modelPackageCatalog = ref({ trained: [], imported: [] });
        const modelPackageUploadRef = ref(null);
        const optModelName = ref("");
        const optAlgorithm = ref("NSGA-II");
        const optCompareAlgorithms = ref([]);
        const optPopulation = ref(64);
        const optGenerations = ref(30);
        const optFormat = ref("png");
        const optDpi = ref(300);
        const optShrinkageModelName = ref("");
        const optShrinkageTimeFeature = ref("");
        const optShrinkageTimePoints = ref("1, 3, 7, 14, 28, 56, 90, 180, 365");
        const optShrinkageTopN = ref(5);
        const optShrinkageProbabilistic = ref(true);
        const optConstraintsText = ref('strength_pred >= 35\nflow_pred >= 180');
        const optSurrogates = ref([]);
        const optFormulaObjectives = ref([]);
        const optVariables = ref([]);
        const optimizationLoading = ref(false);
        const optimizationResult = ref(null);
        const figureUploadRef = ref(null);
        const figurePanels = ref([]);
        const figurePreset = ref('Auto Grid');
        const figureColumns = ref(2);
        const figureCanvasWidth = ref(7.2);
        const figureCanvasHeight = ref(5.4);
        const figureLabelSize = ref(16);
        const figureLabelStyle = ref('lower');
        const figureGap = ref(0.025);
        const figureMargin = ref(0.05);
        const figureFormat = ref('png');
        const figureDpi = ref(300);
        const figureLoading = ref(false);
        const figurePreview = ref('');

        onMounted(() => { console.log("Vue App Mounted Successfully initialized"); });

        const slugify = (value) => String(value || 'plot')
            .trim()
            .replace(/[\\/:*?"<>|]+/g, '_')
            .replace(/\s+/g, '_');

        const inferFormatFromDataUrl = (dataUrl) => {
            if (!dataUrl) return 'png';
            if (dataUrl.startsWith('data:image/svg+xml')) return 'svg';
            if (dataUrl.startsWith('data:image/emf')) return 'emf';
            if (dataUrl.startsWith('data:application/pdf')) return 'pdf';
            return 'png';
        };

        const triggerDownload = (dataUrl, filename) => {
            if (!dataUrl) {
                alert('当前没有可导出的图像。');
                return;
            }
            const anchor = document.createElement('a');
            anchor.href = dataUrl;
            anchor.download = filename;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
        };

        const downloadRenderedPlot = (dataUrl, filenameBase, format) => {
            const finalFormat = format || inferFormatFromDataUrl(dataUrl);
            triggerDownload(dataUrl, `${slugify(filenameBase)}.${finalFormat}`);
        };

        const downloadBase64File = (contentBase64, filename, mimeType) => {
            const binary = atob(contentBase64);
            const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
            const blob = new Blob([bytes], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const anchor = document.createElement('a');
            anchor.href = url;
            anchor.download = filename;
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
            URL.revokeObjectURL(url);
        };

        const exportPlotDataExcel = async (context, plotName, payload = {}) => {
            try {
                const data = await postJson('/api/export/plot_data', { context, plot_name: plotName, payload });
                downloadBase64File(
                    data.content_base64,
                    data.filename || `${slugify(context + '_' + plotName)}_data.xlsx`,
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                );
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const downloadZipBundle = async (bundleName, items) => {
            const validItems = (items || []).filter((item) => item && item.data_url);
            if (!validItems.length) {
                alert('当前页还没有可导出的图。');
                return;
            }
            try {
                const data = await postJson('/api/export/zip', {
                    bundle_name: bundleName,
                    items: validItems
                });
                downloadBase64File(data.content_base64, data.filename || `${slugify(bundleName)}.zip`, 'application/zip');
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const exportCurrentEdaZip = async () => {
            if (!fileData.value) {
                alert('请先上传数据集。');
                return;
            }
            edaLoading.value = true;
            try {
                const data = await postJson('/api/eda/all', {
                    format: edaFormat.value,
                    dpi: Number(edaDpi.value)
                });
                downloadBase64File(data.content_base64, data.filename || 'EDA_All_Feature_Plots.zip', 'application/zip');
                if (data.failed_count > 0) {
                    alert(`已生成 ${data.count} 张图；有 ${data.failed_count} 张因数据类型、缺失值或图形要求不匹配而跳过，详情见压缩包内 EDA_generation_failures.txt。`);
                }
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                edaLoading.value = false;
            }
        };

        const exportCurrentPreprocessZip = async () => {
            await downloadZipBundle('Preprocess_All_Plots', [
                prepOutlierPlot.value ? { name: 'Outlier_Plot', data_url: prepOutlierPlot.value } : null,
                prepScalePlot.value ? { name: 'Scaling_Plot', data_url: prepScalePlot.value } : null,
                prepAugmentPlot.value ? { name: 'Augment_Plot', data_url: prepAugmentPlot.value } : null,
                ...Object.entries(prepAugmentDiagnostics.value || {}).map(([name, dataUrl]) => ({ name: `Synthetic_Diagnostics_${name}`, data_url: dataUrl })),
                prepFeaturePlot.value ? { name: 'Feature_Plot', data_url: prepFeaturePlot.value } : null,
            ]);
        };

        const exportCurrentModelingZip = async () => {
            const items = [
                ...Object.entries(globalPlots.value || {}).map(([name, dataUrl]) => ({ name: `Global_${name}`, data_url: dataUrl })),
                ...Object.entries(trainPlots.value || {}).map(([name, dataUrl]) => ({ name: `${plotSelectedModel.value || 'Model'}_${name}`, data_url: dataUrl })),
            ];
            await downloadZipBundle('Modeling_All_Plots', items);
        };

        const exportCurrentInterpretationZip = async () => {
            const items = [
                ...Object.entries(shapPlots.value || {}).map(([name, dataUrl]) => ({ name: `SHAP_${name}`, data_url: dataUrl })),
                ...Object.entries(advancedXaiPlots.value || {}).map(([name, dataUrl]) => ({ name: `Advanced_XAI_${name}`, data_url: dataUrl })),
                ...Object.entries(pdpPlots.value || {}).map(([name, dataUrl]) => ({ name: `PDP_${name}`, data_url: dataUrl })),
                ...Object.entries(permutationPlots.value || {}).map(([name, dataUrl]) => ({ name: `Permutation_${name}`, data_url: dataUrl })),
                ...Object.entries(counterfactualResult.value?.plots || {}).map(([name, dataUrl]) => ({ name: `Counterfactual_${name}`, data_url: dataUrl })),
                causalResult.value?.graph ? { name: 'Causal_Graph', data_url: causalResult.value.graph } : null,
                causalResult.value?.forest ? { name: 'Causal_Forest', data_url: causalResult.value.forest } : null,
                causalResult.value?.ranking ? { name: 'Causal_Ranking', data_url: causalResult.value.ranking } : null,
                causalResult.value?.estimator_heatmap ? { name: 'Causal_Estimator_Heatmap', data_url: causalResult.value.estimator_heatmap } : null,
                causalResult.value?.dose_response ? { name: 'Causal_Dose_Response', data_url: causalResult.value.dose_response } : null,
            ];
            await downloadZipBundle('Interpretation_All_Plots', items);
        };

        const exportCurrentOptimizationZip = async () => {
            const items = [
                ...Object.entries(optimizationResult.value?.plots || {}).map(([name, dataUrl]) => ({ name: `Optimization_${name}`, data_url: dataUrl })),
                ...Object.entries(optimizationResult.value?.comparison_plots || {}).map(([name, dataUrl]) => ({ name: `Optimization_Compare_${name}`, data_url: dataUrl })),
            ];
            await downloadZipBundle('Optimization_All_Plots', items);
        };

        const exportCurrentFigureZip = async () => {
            await downloadZipBundle('Figure_Studio_All_Plots', [
                figurePreview.value ? { name: 'Figure_Studio_Preview', data_url: figurePreview.value } : null
            ]);
        };

        const exportModelPlotCategoryOverview = async (category, formatOverride = null) => {
            if (!category || !category.plots || !category.plots.length) {
                alert('该分类下没有可拼接的图。');
                return;
            }
            const fmt = formatOverride || categoryOverviewFormat.value || 'png';
            const panels = category.plots.map((plot, idx) => ({
                data_url: plot.dataUrl,
                name: plot.name,
                label: String.fromCharCode(97 + (idx % 26)),
                show_label: true,
            }));
            try {
                const data = await postJson('/api/figure/compose', {
                    panels,
                    canvas_width: Number(categoryOverviewWidth.value) || 7.2,
                    canvas_height: Number(categoryOverviewHeight.value) || 9.7,
                    columns: Number(categoryOverviewColumns.value) || 2,
                    label_size: 18,
                    label_style: 'lower',
                    gap: 0.025,
                    margin: 0.04,
                    preset: 'Auto Grid',
                    format: fmt,
                    dpi: Number(categoryOverviewDpi.value) || 300,
                });
                downloadRenderedPlot(data.image, `${plotSelectedModel.value || 'Model'}_${category.label}_Overview`, fmt);
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const plotHelpText = {
            'Model Leaderboard': '先看整体排名。回归看 R2，分类看 Accuracy，越高越好，但它本身不能判断是否过拟合。',
            'Multi-Model Taylor Diagram': '这是少量核心模型的综合对比图。越接近 Reference 越好，相关系数越高越好，标注的 d 越小越好。',
            'Multi-Model Performance Matrix': '这是多指标热图。重点一起看 Test 分数、Train 分数、CV Mean 和 Gap Penalty，判断谁既强又稳。',
            'Overfitting Risk Map': '这张图最适合排查过拟合。越靠左说明 train/test 差距越小，越靠上说明测试表现越强，左上角最理想。',
            'Train-Test Generalization Ladder': '每个模型一条训练到测试的连线。线越短越稳，线越长越可能过拟合。',
            'Consensus Model Rank Matrix': '把多个指标转成排名，综合判断模型是否稳定靠前，而不是只看单一分数。',
            'Model Capability Radar': '用雷达图同时展示模型表现、稳定性和泛化能力，适合筛选综合最均衡的模型。',
            'Model Selection Pareto Frontier': '在测试表现和泛化差距之间找 Pareto 前沿模型，用于避免选到高分但过拟合的模型。',
            'Model Prediction Correlation Matrix': '比较不同模型预测结果是否高度一致。高度一致说明模型族结论稳定，差异大说明存在模型不确定性。',
            'Model Prediction Agreement Matrix': '分类任务中比较模型预测标签一致性，用于发现不稳定类别判断。',
            'Ensemble Disagreement Error Analysis': '比较多个模型在同一样本上的预测分歧是否对应更大误差。分歧越大误差越大时，可把分歧作为可信度指标。',
            'Ensemble Vote Confidence Analysis': '分类任务中检查模型投票置信度和实际准确率的关系，用于判断集成共识是否可信。',
            'Model Error Signature Map': '按样本展示不同模型的误差签名，能发现所有模型共同困难的样本或特定模型失效模式。',
            'Wilcoxon Significance Analysis': '用 Wilcoxon 符号秩检验比较当前最佳模型与其他模型的逐样本误差差异，并显示 BH 校正后的显著性结果。',
            'Cohens D Effect Size': '量化最佳模型和其他模型之间误差差异的实际大小。绝对值越大，说明差异在实际意义上越明显。',
            'Prediction vs True': '看预测值是否贴近对角线。越贴近越好；如果高值区或低值区明显偏离，说明该区间拟合不足。',
            'Residuals Plot': '看残差是否围绕 0 随机散开。若有漏斗形或弯曲结构，通常说明异方差或遗漏了非线性关系。',
            'Residuals Density': '看测试残差整体是否集中在 0 附近。越尖越窄通常越好，偏斜则说明有系统偏差。',
            'Residual Q-Q Plot': '看残差是否近似正态。越接近对角线越理想，两端偏离大说明极端误差更重。',
            'Bland-Altman Plot': '看预测偏差和一致性范围。中心偏差越接近 0 越好，上下限越窄越稳定。',
            'Train vs Test R2': '直接看训练集和测试集差多少。训练高很多而测试低，就是典型过拟合信号。',
            'Generalization Summary': '把 Train/Test 的 R2、RMSE、MAE 放在一起看。不要只看 R2，误差指标也要一起判断。',
            'Train-Test Residual Density': '比较训练集和测试集残差分布。两条曲线越接近，泛化越稳定。',
            'Absolute Error ECDF': '看绝对误差累计分布。曲线越靠左，说明大多数样本误差越小。',
            'Residual Split Comparison': '把训练和测试残差放在一张图里比较。测试点云如果明显更散，通常泛化较差。',
            'Residuals vs Observed': '看误差是否随真实值区间系统偏移。如果高值或低值区域残差整体偏向一侧，说明分段拟合存在偏差。',
            'Quantile Error Profile': '按目标值分位数看预测均值和误差。适合判断模型在哪些数值区间表现最好、哪些区间最吃力。',
            'Residual-Feature Bias Scan': '逐个扫描关键特征上的残差结构。如果某个特征区间残差系统偏正或偏负，说明模型在该区间有偏差。',
            'Error Hotspot Map': '用两个最相关的特征分箱定位高误差区域，适合找模型最容易失效的材料配比或参数组合。',
            'Feature Quantile Error Matrix': '按特征分位区间统计平均绝对误差，用于定位哪些特征区间最容易产生大误差。',
            'Worst Error Waterfall': '展示误差最大的样本及偏差方向，适合回溯异常样本或模型外推失败案例。',
            'Residual Run and Autocorrelation': '检查残差是否随样本顺序呈现系统漂移或自相关，发现批次效应和排序相关偏差。',
            'Feature Slice Bias Heatmap': '按多个特征的分位区间统计平均残差，定位模型系统性高估或低估的特征切片。',
            'Conformal Prediction Diagnostics': '使用训练残差构造共形预测区间，并检查测试覆盖率是否达到名义水平。',
            'Probabilistic Prediction Interval': '使用支持 ensemble 的模型成员预测构造概率区间。紫色带是 5%-95% 预测区间，真实值落入区间的比例代表覆盖率，右图检查区间宽度、覆盖率和误差是否一致。',
            'Uncertainty Error Alignment': '检查模型给出的不确定性是否真的对应更大的误差，以及预测区间覆盖是否稳定。',
            'Prediction Risk Stratification': '结合适用域距离、不确定性宽度等信号构建风险分层，检查高风险样本是否确实误差更大。',
            'Permutation Interaction Screen': '通过联合置换筛查特征交互强度，帮助发现模型依赖的关键二阶交互。',
            'Prediction Diagnostic Gallery': '汇总预测密度、绝对误差 ECDF、Scale-Location 和残差秩签名，快速判断误差分布形态。',
            'Residual Density Hexbin': '用六边形密度图观察预测值-残差分布，适合大样本下发现系统偏差结构。',
            'Target-Segment Performance Dashboard': '按目标值区间统计 R2、RMSE、MAE 和 Bias，判断模型在哪些目标区间表现不足。',
            'Observed-Predicted Distribution Shift': '比较训练/测试的观测值、预测值和残差分布是否发生漂移。',
            'One-Factor Response Sensitivity Grid': '固定其他变量，逐个扫描特征变化对模型预测的影响，适合展示模型响应方向。',
            'Regression Calibration Belt': '按预测值分箱比较平均预测和平均观测，并给出置信带，判断回归模型是否校准。',
            'Error Violin by Target Segment': '按目标值区间展示误差分布形态，判断模型在低值、中值、高值区域是否稳定。',
            'Standardized Residual Control Chart': '用控制图识别超出 2/3 sigma 的异常残差，适合发现不可忽略的失控样本。',
            'Residual Statistics Panel': '汇总残差均值、偏度、峰度和尾部误差，快速判断残差分布是否偏斜或重尾。',
            'Observed-Predicted Joint Marginal Map': '联合展示观测值和预测值的密度及边际分布，适合论文中展示拟合结构。',
            'Error Exceedance Curve': '展示绝对误差超过不同阈值的概率，用于说明误差尾部风险。',
            'Residual Lag Scatter': '检查相邻样本残差是否存在连续结构或自相关，适合发现排序、批次或时间相关偏差。',
            'Rolling RMSE Trace': '沿样本顺序滚动计算 RMSE，用于定位模型性能在局部样本段中的退化区域。',
            'Prediction Error Bubble Map': '用预测值、真实值和误差气泡大小同时展示误差热点，适合快速识别高风险预测区。',
            'Absolute Error Pareto Chart': '按样本绝对误差从大到小排序，展示少数高误差样本对总体误差的贡献。',
            'Error Decile Lift Chart': '比较不同预测分位段的平均误差和最大误差，用于发现预测区间内的系统性风险。',
            'Target Coverage Error Band': '按真实值排序展示预测误差带，检查模型在低值、中值和高值区域的覆盖质量。',
            'Prediction-Segment Bias with CI': '按预测分段统计平均残差及置信区间，判断模型是否在某些预测区间稳定高估或低估。',
            'Residual Normality Detail Panel': '联合展示残差直方图、Q-Q 图和尾部指标，用于判断正态性、偏斜和重尾问题。',
            'Signed Residual ECDF': '分别展示正残差和负残差的经验累积分布，评估高估与低估的尾部差异。',
            'Error Symmetry Plot': '比较正负残差幅度分布，辅助判断误差是否围绕零对称。',
            'Cumulative Error Contribution Curve': '展示误差从大到小累积贡献比例，量化头部困难样本对整体误差的影响。',
            'Prediction Rank Concordance': '比较真实值排序和预测值排序的一致性，适合评估模型对样本相对顺序的保持能力。',
            'Local Error Volatility': '沿样本顺序展示滚动误差标准差，用于定位模型不稳定区间。',
            'Residual Periodogram Spectrum': '分析残差频谱，帮助发现周期性、批次性或隐藏结构未被模型解释。',
            'Bland-Altman Density Map': '用 Bland-Altman 形式展示均值水平与预测差异，适合评估一致性和系统偏倚。',
            'Calibration Residual Belt': '按预测值分段展示残差均值和标准差带，判断预测尺度上的校准偏差。',
            'Error Boxen by Prediction Segment': '用箱线分布比较不同预测区间的绝对误差，识别误差异方差和异常段。',
            'Relative Error Distribution': '展示相对误差百分比分布，适合目标量纲较大或跨尺度数据的误差解释。',
            'MAPE by Target Segment': '按真实值分段统计 MAPE，用于判断模型在哪些目标区间相对误差更高。',
            'Input Noise Robustness Curve': '对输入特征加入不同强度扰动并观察预测变化，评估模型对输入噪声的鲁棒性。',
            'Feature-Error Correlation Bars': '计算特征与绝对误差的相关性，筛选可能导致误差升高的关键输入变量。',
            'Error Tolerance Utility Curve': '展示不同误差容忍阈值下可接受样本比例，用于把模型误差转化为应用可用性判断。',
            'Residual Tail QQ Plot': '重点标记残差分布两端尾部，判断极端误差是否偏离正态假设。',
            'Residual Sign Transition Matrix': '统计相邻样本残差符号转移，识别连续高估或连续低估的结构性模式。',
            'Residual Quantile Trend': '展示滚动残差 P10、P50、P90 变化，用于发现局部偏移和尾部误差漂移。',
            'Prediction Quantile Calibration Ladder': '按预测分位比较真实均值与预测均值，评估不同预测区间的校准质量。',
            'Target Quantile Bias Ladder': '按真实目标分位统计中位预测偏差，发现低值或高值区间的系统性偏差。',
            'Hard-Sample Fingerprint Heatmap': '对误差最大的样本展示特征标准化画像，用于定位困难样本的共同特征组合。',
            'Easy-vs-Hard Feature Shift': '比较高误差样本和低误差样本的特征均值差异，筛选导致模型失效的候选变量。',
            'Residual Leverage Proxy Map': '把样本投影到特征潜在空间并用误差着色，观察高误差样本是否集中在特殊区域。',
            'Distance-to-Center Error Curve': '分析样本离特征中心越远时误差是否上升，用于评估外推风险。',
            'Neighbor Error Smoothness': '比较样本误差与邻近样本平均误差，判断错误是否呈局部聚集。',
            'Two-Feature Error Surface': '在两个关键特征构成的网格中显示平均绝对误差，定位交互误差热点。',
            'Feature-Binned Residual Heatmap': '在双特征分箱中显示平均残差，识别特定特征组合下的高估或低估。',
            'Feature Outlier Error Risk': '比较每个特征异常值样本与非异常样本的误差风险比，评估异常输入对模型的影响。',
            'Conditional Coverage by Feature Segment': '统计不同特征分段下误差容忍覆盖率差异，检查模型在各输入区域是否公平稳定。',
            'Prediction Density Residual Rug': '叠加预测值密度和残差散点，展示模型主要预测区域内的误差分布。',
            'Robust Loss Sensitivity Curve': '展示不同 Huber 阈值下的稳健损失变化，评估模型误差对异常点的敏感性。',
            'Error Severity Composition': '把误差分为低、中、高、极端等级，直观展示模型输出风险结构。',
            'Worst-Case Error Frontier': '展示最差样本尾部的平均误差，量化模型在极端失败场景下的表现。',
            'Signed Error Waterfall': '按绝对误差排序展示带方向的预测误差，区分严重高估和严重低估。',
            'Error Tail Concentration Donut': '展示前 10% 最差样本贡献的误差占比，用于说明错误是否集中在少数样本。',
            'Percentile Calibration Map': '比较预测百分位和真实百分位的一致性，评估模型对样本排序和分位校准的保持能力。',
            'Threshold Utility Diagnostics': '分类任务中比较不同阈值下 Precision、Recall、F1 和阳性率，用于选择业务或论文中可解释的决策阈值。',
            'K-Fold Stability': '看模型在不同折划分下分数波动是否稳定。均值高且波动小，说明模型更可靠。',
            'Y-Randomization Test': '把训练标签打乱后重新建模，若随机化分数远低于真实模型，说明性能更可能来自真实信号。',
            'Williams Plot': '通过杠杆值和标准化残差识别异常点与适用域，超出阈值的点需要重点检查。',
            'Bootstrap Stability': '看同一模型在多次 bootstrap 重采样下分数分布是否稳定。均值高且分布窄，说明模型稳健性更强。',
            'Bootstrap Uncertainty Band': '看 bootstrap 预测区间包络。区间越窄说明不确定性越低；若真实值频繁落在区间外，说明模型稳定性不足。',
            'Applicability Domain Distance': '比较训练域和测试样本到训练邻域的距离分布。测试样本若大量越过阈值，说明外推风险较高。',
            'Applicability Domain Reliability': '看“离训练域越远，误差是否越大”。若呈明显上升趋势，说明适用域边界外预测不可靠。',
            'Applicability Domain Coverage': '总结测试样本有多少落在适用域内。适用域内比例越高，模型在当前数据上的可信覆盖越好。',
            'Interval Stability Analysis': '按目标值区间比较训练集和测试集的 MAE，适合看模型在哪些数值段更稳定。',
            'Interval Stability Radial': '用期刊风格的径向图比较不同目标区间的 Train/Test MAE，越外侧表示该区间误差更大。',
            'Hyperparameter Search History': '看超参数优化过程中分数如何变化，以及 n_estimators/max_depth 的搜索轨迹。',
            'Hyperparameter Search Heatmap': '看不同 n_estimators 和 max_depth 组合下的分数高低，帮助判断高分区域是否稳定成片出现。',
            'Hyperparameter Response Surface': '用响应面看调参空间的整体地形。高分峰如果很尖，说明参数较敏感；高分平台更说明模型稳。',
            'Top Hyperparameter Trials': '列出当前搜索过程中得分最好的参数组合，适合快速挑候选配置做复验。',
            'Meta Score Distribution': '展示元启发式搜索过程中所有候选参数的得分分布，判断搜索是否稳定。',
            'Meta Improvement Curve': '展示最优分数随迭代的增益，判断优化是否真正带来改进。',
            'Meta Exploration Exploitation': '同时查看参数移动幅度和得分增益，判断搜索是在探索还是局部微调。',
            'Meta Algorithm Model Score Matrix': '比较不同元启发式算法与不同模型组合的最佳 CV 分数，例如 PSO-XGBoost、GA-LightGBM。',
            'Meta-Model Combination Leaderboard': '把所有元启发式-模型组合按最佳分数排序，直接选择最强组合。',
            'Meta Convergence Comparison': '比较多个元启发式-模型组合的收敛过程，看谁收敛快、谁最终分数高。',
            'Meta Search Landscape Scatter': '展示元启发式搜索过的参数点和对应分数，判断搜索空间是否被充分探索。',
            'Meta Stability and Efficiency': '比较不同元启发式算法的得分稳定性和单位迭代效率。',
            'Overfitting Fingerprint': '这是浓缩版过拟合指纹图。重点看 Gap 和 RMSE Ratio，越大通常越危险。',
            'Learning Curve': '看样本量增加时 Train/Validation 如何变化。长期分离是过拟合，两条都低是欠拟合，逐渐靠拢且较高更理想。',
            'Feature Importance': '看模型主要依赖哪些特征，但它不是因果结论，更像贡献度线索。',
            'Series Overlay': '抽取一段样本做真实值和预测值叠加，对比峰值、谷值和局部波动是否跟得上。',
            'Confusion Matrix': '看分类任务每一类被分对还是分错。对角线越强越好。',
            'Normalized Confusion Matrix': '看各类别内部的命中率，更适合样本不平衡时判断哪一类最难。',
            'Class-wise Error Profile': '逐类别查看 Precision、Recall、F1 和样本数，避免总体准确率掩盖少数类失效。',
            'ROC Curve': '二分类看整体区分能力，越靠左上越好，AUC 越大越强。',
            'Precision-Recall Curve': '当正负样本不平衡时更有参考价值，曲线越高越好。',
            'Calibration Curve': '看预测概率是否可信，越贴近对角线说明概率越校准。',
            'Probability Separation': '看正负样本概率分布分得开不开，重叠越少越好。',
            'Cumulative Gain Curve': '看按模型置信度排序后，能多快抓住正样本。曲线越早抬升越好。',
            'KS Diagnostic Curve': '看正负样本分布分离程度，KS 越大通常区分能力越强。',
            'Train vs Test Accuracy': '分类版的训练/测试对比，差距大时通常要警惕过拟合。',
            'Metric Comparison': '分类下同时看 Accuracy、Precision、Recall、F1 的训练与测试差异。',
            'Generalization Gap by Metric': '逐项看分类指标的 train-test 差值，越接近 0 越稳。',
            'Generalization Scorecard': '分类任务的综合热图，适合快速看哪项指标最短板。',
            'Confidence Shift': '比较训练集和测试集的预测置信度分布，若测试明显更分散，说明泛化稳定性一般。',
            'Permutation Cumulative Drop Curve': '看前几个重要特征累计贡献了多少性能变化，能帮助你判断解释是否被极少数特征主导。',
            'SHAP Interaction Proxy Heatmap': '看关键特征的 SHAP 贡献是否一起变化，可作为潜在交互关系的线索图。',
            'SHAP Dependence Panel': '同时展示多个关键特征的取值和 SHAP 贡献关系，用于判断特征在哪些区间推动或抑制预测。',
            'SHAP Cohort Heatmap': '把样本按解释强度排序，观察不同样本群体是否由不同特征主导。',
            'SHAP Attribution Concentration': '看模型解释是否被极少数特征主导，适合判断结论是否过度集中。',
            'Designed SHAP Beeswarm Density': '重新设计的 SHAP 蜂群密度图。横轴是贡献方向和大小，颜色表示特征值高低，点大小强调强贡献样本；适合一眼看出哪些变量在什么取值下推动预测升高或降低。',
            'SHAP Interaction Constellation': '特征归因星座图。节点越大说明该特征总体解释力越强，连线表示两个特征的 SHAP 贡献协同或对抗，用于发现潜在交互关系。',
            'Local Explanation Storyboard': '单个样本的解释故事板。从基线预测开始，按贡献最大的特征逐步累加到最终预测，右侧列出该样本的关键特征值和推动方向。',
            'Attribution Terrain Map': '把所有样本的解释向量压缩到二维地形图。背景表示预测水平，点大小和颜色表示解释强度，可用于发现不同机理区域和异常解释簇。',
            'Local Explanation Fingerprints': '局部解释指纹图。比较指定样本、典型样本和强解释样本的归因形状，判断某个样本是否属于常规决策模式。',
            'Prediction Reason Wheel': '最直观的单样本解释图。圆环表示预测主要由哪些因素构成，红色推动预测升高、蓝色推动预测降低，中间是最终预测值。',
            'Plain-Language Factor Cards': '把模型解释转成类似诊断卡片的形式：该变量在样本中是高值、中值还是低值，以及它把预测推高还是压低。',
            'What-If Response Ladder': '直接回答“如果改这个变量会怎样”。对关键特征分别替换为低位数、中位数和高位数，展示预测会升高还是降低。',
            'Similar-Case Explanation Contrast': '把当前样本和最相似的训练样本放在一起比较，显示它们预测值和关键特征贡献是否一致，用于判断这个解释是否孤立。',
            'Mechanism Quadrant Map': '把每个关键特征分成“低值时的作用”和“高值时的作用”。右上表示高低都推高，左下表示高低都压低，对角象限表示方向随取值改变。',
            'SHAP Directional Importance': '把每个特征的正向和负向平均贡献分开看，判断特征主要推动预测升高还是降低。',
            'SHAP Positive-Negative Dominance': '比较正向贡献和负向贡献强度，识别方向性明确或双向作用明显的特征。',
            'SHAP Contribution Variance': '展示各特征 SHAP 值的样本间方差，方差越大说明该特征作用更依赖具体样本背景。',
            'SHAP Attribution Entropy Distribution': '衡量每个样本的解释是否集中在少数特征上。熵低表示少数特征主导，熵高表示多特征共同作用。',
            'SHAP Top Driver Frequency': '统计每个特征成为样本主导解释因素的次数，适合发现最常控制个体预测的变量。',
            'SHAP Prediction-Cohort Profile': '按预测值分组展示平均 SHAP 贡献，判断低预测和高预测样本是否由不同特征驱动。',
            'SHAP Attribution Correlation Network': '查看不同特征的 SHAP 贡献是否同步变化，可作为模型内部交互或冗余解释的线索。',
            'SHAP Sample Influence Concentration': '展示少数样本是否贡献了大部分解释强度，帮助识别解释是否被异常样本主导。',
            'Highest-Influence Sample Decomposition': '选择解释强度最高的样本，展示其各特征正负贡献，适合做局部案例解释。',
            'SHAP Value-Bin Effect Matrix': '按特征取值分箱统计平均 SHAP，帮助把复杂解释转成可读的区间效应矩阵。',
            'XAI Explanation Manifold': '把每个样本的 SHAP 解释向量降维成二维流形，并用预测值着色，用于发现解释模式是否形成清晰样本群。',
            'XAI Manifold by Dominant Feature': '同样是解释流形，但按主导解释特征着色，能看出不同机制样本是否在解释空间中分离。',
            'SHAP Contribution River': '按预测值排序后堆叠展示正负 SHAP 贡献，适合做视觉冲击强的机制流图，展示模型预测从低到高时由哪些特征推动。',
            'SHAP Polar Attribution Rose': '用极坐标玫瑰图展示全局 SHAP 贡献占比，适合作为论文中更美观的全局解释概览图。',
            'Local Explanation Similarity Map': '计算样本解释向量之间的相似性，判断模型是否存在稳定的解释亚型或异常解释样本。',
            'Explanation Cluster Heatmap': '把样本按 SHAP 解释聚类后展示热图，用于发现不同解释机制群体。',
            'Explanation Cluster Signature Radar': '为每个解释聚类绘制雷达图，比较不同样本群的主导特征签名。',
            'Prototype-Criticism Explanation Map': '在解释流形中标记代表性原型样本和边缘反例样本，适合挑选典型案例和异常案例。',
            'Bootstrap Attribution Stability': '对样本重采样后计算特征贡献置信区间，判断解释排序是否稳定。',
            'Surrogate Rule Tree': '用浅层决策树近似复杂模型预测，生成可读规则结构，适合补充材料说明模型决策路径。',
            'Representative Local Explanation Gallery': '展示多个代表性样本的局部贡献条形图，适合论文中说明不同样本的解释差异。',
            'Explanation Strength vs Prediction': '比较预测值和解释强度，判断高预测或低预测样本是否需要更强的特征贡献支撑。',
            'ALE Main Effect': 'ALE 比 PDP 更不容易受强相关特征误导。曲线表示该特征局部变化对预测的净影响方向和强度。',
            'ALE Density Overlay': '把特征分布和 ALE 放在一起看，能判断模型结论主要来自高密度数据区还是稀疏区。',
            'ICE Heterogeneity and Sensitivity': '展示同一特征对不同样本的局部响应差异，以及模型最敏感的特征区间。',
            'Marginal Response Derivative': '展示模型预测对所选特征的局部斜率，斜率绝对值越大说明模型在该区间越敏感。',
            'Response Monotonicity Diagnostics': '同时展示参考响应曲线和斜率方向占比，用于判断模型响应是否单调、是否存在反转。',
            'Quantile-Anchored Response Curves': '在不同样本背景分位数下扫描同一特征，判断特征效应是否依赖其他变量背景。',
            'Response Extrapolation Risk': '把响应曲线与特征密度结合，提示哪些特征区间数据稀疏、解释更可能属于外推。',
            'Counterfactual Threshold Path': '展示为了达到不同预测水平，所选特征大约需要达到的值，适合做可操作设计建议。',
            'Pairwise Response Interaction Ridge': '展示两个特征共同变化时的预测响应面，用于发现协同、拮抗或非线性交互。',
            'Conditional Interaction Strength Curve': '量化一个特征效应随另一个特征变化的强弱，适合筛选关键交互区间。',
            'Permutation Repeat Distribution': '展示置换重要性在多次重复中的分布，分布越稳定说明特征重要性越可靠。',
            'Permutation Importance Signal-to-Noise': '用均值除以标准差衡量置换重要性的信噪比，区分稳定重要特征和不稳定特征。',
            'Permutation Rank Stability': '展示特征重要性排名在重复置换中的均值和波动，排名越靠前且波动越小越可信。',
            'Permutation Redundancy Proxy Map': '结合特征相关性和置换重要性，识别可能互相替代或冗余的关键特征组。',
            'Top-k Feature Sufficiency Proxy': '只保留前 k 个重要特征并打乱其余特征，观察预测方差保留程度，评估少数特征是否足够解释模型。',
            'Counterfactual Feature Shifts': '看为了把预测推向目标值，各特征需要朝哪个方向改、改多少。',
            'Counterfactual Comparison Heatmap': '把原始样本和代表性反事实样本并排比较，适合快速看关键变量组合变化。',
            'Counterfactual Prediction Shift': '看原始预测和各个反事实预测的变化幅度，判断哪些方案更接近期望目标。'
        };

        const getPlotHelp = (plotName) => plotHelpText[plotName] || '这张图用于补充判断模型表现，建议结合排行榜、泛化图和残差图一起看。';

        const publicationPlotHelpText = {
            'Leakage Correlation Sentinel': '数据泄露哨兵图。每一条代表一个特征与目标值的绝对相关性；越接近 1，越需要警惕该特征是否直接或间接包含了目标信息。若出现 |corr| ≥ 0.95 的特征，投稿前应人工确认它不是由目标计算得来、不是后验测量值，也没有跨训练/测试泄露。',
            'Ablation Performance Drop': '消融实验图。横轴是移除某个特征后模型指标相对基线的变化，通常回归看 R2、分类看 Accuracy。数值越负，说明移除该特征后性能下降越明显，该特征对模型越关键；如果移除后性能反而提升，说明该特征可能带来噪声或过拟合。',
            'External Validation Observed vs Predicted': '外部验证拟合图。横轴是真实值，纵轴是模型在独立外部数据上的预测值；点越贴近对角线，说明模型在外部数据上泛化越好。若外部验证明显偏离测试集表现，说明模型可能依赖内部数据分布，论文中需要谨慎表述。',
            'Repeated Validation Stability': '重复交叉验证稳定性图。每个点是一次重复交叉验证折的分数，虚线是平均分，阴影是标准差范围。曲线波动越小，说明模型对数据划分越不敏感；如果波动很大，单次 train/test 分数不适合作为强结论。',
            'Subgroup Failure Discovery': '分群失效发现图。系统自动按特征区间寻找误差显著升高的样本群，横轴是该分群的误差相对总体平均误差的倍数。大于 1 表示该区间比平均更容易出错；最高的几个分群应作为模型局限、适用边界或补充实验对象。',
            'Train-Test Domain Shift': '训练-测试域漂移图。每个条形表示某个特征在测试集和训练集之间的标准化分布差异，越大说明测试集越偏离训练域。高漂移特征可能导致外推风险，论文中可用它解释为什么某些样本误差较大。',
            'Active Learning Priority Map': '主动学习优先级图。横轴是样本在特征空间中相对训练集的新颖性，纵轴是误差或错误风险；右上角样本既新颖又高风险，最适合优先补实验、复测或加入下一轮数据采集。',
            'Symbolic Surrogate Terms': '符号代理项图。它用一个简单的二阶稀疏公式去近似复杂模型的预测行为，条形表示各公式项的贡献方向和大小。正值推动预测升高，负值推动预测降低；它不是因果公式，但适合生成可读的机理假设和补充材料公式。',
        };

        const getPublicationPlotHelp = (plotName) => publicationPlotHelpText[plotName] || '这是论文证据链辅助图。建议结合对应表格、外部验证和领域知识判断，不要只凭单张图下结论。';

        const generationPlotHelpText = {
            'Synthetic Sample and Target Balance': '检查增强前后样本量和目标分布是否合理。分类任务应避免类别比例被异常扭曲；回归任务应避免目标分布出现不真实峰值或长尾。',
            'Synthetic Feature Mean Shift': '展示增强后特征均值相对原始数据的标准化偏移。偏移越大，说明生成数据可能改变了原始数据中心，应重点检查对应特征。',
            'Synthetic Variance Preservation': '比较增强后与原始数据的标准差比例。接近 1 表示方差保真较好；远大于 1 或远小于 1 表示生成数据可能过度扩散或过度收缩。',
            'Synthetic Distribution Drift Sentinel': '用 ECDF 距离衡量每个特征增强前后的分布漂移。数值越高，说明该特征的生成分布与原始分布差异越明显。',
            'Synthetic Correlation Structure Delta': '比较增强前后特征相关矩阵的变化。大面积红蓝块说明生成算法改变了变量之间的依赖结构。',
            'Synthetic PCA Coverage Map': '把原始样本和合成样本投影到 PCA 空间。理想情况是合成样本覆盖原始数据邻域，而不是形成远离原始数据的孤岛。',
            'Synthetic Nearest-Neighbor Distance': '统计合成样本到最近原始样本的距离。距离过小可能是记忆/复制，距离过大可能是不可信外推。',
            'Synthetic Memorization Risk Curve': '按最近邻距离排序识别最贴近原始样本的合成点。曲线底部过低时，要警惕生成样本只是复制原始样本。',
            'Synthetic Target Relationship Drift': '检查增强后特征与目标值关系是否被改变。漂移大的特征可能导致模型学到不真实关系。',
            'Synthetic Density Overlay': '对漂移最大的特征叠加原始和增强密度曲线，直观看生成分布是否合理。',
            'Synthetic Covariance Spectrum': '比较协方差特征值谱，判断增强是否保留了数据的整体多变量结构和主变化方向。',
            'Synthetic Data Quality Scorecard': '汇总均值漂移、分布漂移、相关结构变化和方差偏差，作为合成数据质量的快速总览。',
        };

        const getGenerationPlotHelp = (plotName) => generationPlotHelpText[plotName] || '这是合成数据质量诊断图，用于判断生成样本是否保留原始数据分布、相关结构和目标关系。';

        const modelPlotCategoryRules = [
            { id: 'classification', words: ['confusion', 'roc', 'precision-recall', 'calibration curve', 'probability separation', 'cumulative gain', 'ks diagnostic', 'accuracy', 'metric comparison', 'class-wise', 'confidence shift', 'threshold utility'] },
            { id: 'optimization', words: ['hyperparameter', 'meta ', 'meta-', 'search history', 'search heatmap', 'response surface', 'top hyperparameter'] },
            { id: 'overfit', words: ['overfitting', 'generalization gap', 'train vs test r2', 'train-test generalization', 'train vs test accuracy', 'learning curve'] },
            { id: 'generalization', words: ['bootstrap', 'k-fold', 'y-randomization', 'train-test residual', 'residual split', 'observed-predicted distribution shift', 'interval stability', 'generalization summary', 'scorecard', 'stability'] },
            { id: 'calibration', words: ['calibration', 'conformal', 'uncertainty', 'prediction interval', 'coverage', 'percentile calibration'] },
            { id: 'robustness', words: ['applicability domain', 'williams', 'input noise', 'distance-to-center', 'feature outlier', 'conditional coverage', 'robust loss'] },
            { id: 'feature', words: ['feature importance', 'feature-', 'feature ', 'permutation interaction', 'one-factor', 'response sensitivity', 'two-feature', 'easy-vs-hard', 'hard-sample', 'leverage proxy', 'neighbor error'] },
            { id: 'risk', words: ['worst', 'hotspot', 'pareto', 'tail', 'severity', 'exceedance', 'tolerance utility', 'waterfall', 'error frontier', 'error concentration', 'risk stratification', 'bubble map', 'error decile', 'error boxen', 'relative error', 'mape'] },
            { id: 'residual', words: ['residual', 'bland-altman', 'normality', 'qq', 'ecdf', 'symmetry', 'periodogram', 'density hexbin', 'control chart', 'signed error'] },
            { id: 'accuracy', words: ['prediction vs true', 'model leaderboard', 'taylor', 'performance matrix', 'absolute error', 'rank concordance', 'target-segment performance', 'quantile error', 'series overlay'] },
            { id: 'summary', words: ['radar', 'capability', 'consensus', 'selection pareto', 'diagnostic gallery', 'summary', 'matrix', 'leaderboard'] },
        ];

        const getModelPlotCategoryId = (plotName) => {
            const text = String(plotName || '').toLowerCase();
            const hit = modelPlotCategoryRules.find((rule) => rule.words.some((word) => text.includes(word)));
            return hit ? hit.id : 'other';
        };

        const getModelPlotCategory = (plotName) => {
            const id = getModelPlotCategoryId(plotName);
            return modelPlotCategories.find((category) => category.id === id) || modelPlotCategories[modelPlotCategories.length - 1];
        };

        const categorizedModelPlots = computed(() => {
            const entries = Object.entries(trainPlots.value || {});
            return modelPlotCategories
                .map((category) => ({
                    ...category,
                    plots: entries
                        .filter(([name]) => getModelPlotCategoryId(name) === category.id)
                        .map(([name, dataUrl]) => ({ name, dataUrl })),
                }))
                .filter((category) => category.plots.length > 0);
        });

        const visibleCategorizedModelPlots = computed(() => {
            if (selectedModelPlotCategory.value === 'all') return categorizedModelPlots.value;
            return categorizedModelPlots.value.filter((category) => category.id === selectedModelPlotCategory.value);
        });
        const modelPlotCategoryFilters = computed(() => modelPlotCategories.filter((category) => category.id !== 'other'));

        const postJson = async (url, payload) => {
            const res = await axios.post(url, payload);
            if (res.data.status !== 'success') {
                throw new Error(res.data.message || 'Request failed.');
            }
            return res.data;
        };

        const downloadSvgFromImageEndpoint = async ({ url, payload, imageKey = 'image', filenameBase }) => {
            const data = await postJson(url, { ...payload, format: 'svg' });
            const image = data[imageKey];
            if (!image || !String(image).startsWith('data:image')) {
                throw new Error('SVG export unavailable.');
            }
            triggerDownload(image, `${slugify(filenameBase)}.svg`);
        };

        const downloadSvgFromPlotsEndpoint = async ({ url, payload, plotName, filenameBase, plotsKey = 'plots' }) => {
            const data = await postJson(url, { ...payload, format: 'svg' });
            const image = data?.[plotsKey]?.[plotName];
            if (!image || !String(image).startsWith('data:image')) {
                throw new Error(`SVG export unavailable for ${plotName}.`);
            }
            triggerDownload(image, `${slugify(filenameBase || plotName)}.svg`);
        };

        const downloadPdfFromImageEndpoint = async ({ url, payload, imageKey = 'image', filenameBase }) => {
            const data = await postJson(url, { ...payload, format: 'pdf' });
            const file = data[imageKey];
            if (!file || !String(file).startsWith('data:application/pdf')) {
                throw new Error('PDF export unavailable.');
            }
            triggerDownload(file, `${slugify(filenameBase)}.pdf`);
        };

        const downloadPdfFromPlotsEndpoint = async ({ url, payload, plotName, filenameBase, plotsKey = 'plots' }) => {
            const data = await postJson(url, { ...payload, format: 'pdf' });
            const file = data?.[plotsKey]?.[plotName];
            if (!file || !String(file).startsWith('data:application/pdf')) {
                throw new Error(`PDF export unavailable for ${plotName}.`);
            }
            triggerDownload(file, `${slugify(filenameBase || plotName)}.pdf`);
        };

        const downloadEmfFromImageEndpoint = async ({ url, payload, imageKey = 'image', filenameBase }) => {
            const data = await postJson(url, { ...payload, format: 'emf' });
            const file = data[imageKey];
            if (!file || !String(file).startsWith('data:image/emf')) {
                throw new Error('EMF export unavailable.');
            }
            triggerDownload(file, `${slugify(filenameBase)}.emf`);
        };

        const downloadEmfFromPlotsEndpoint = async ({ url, payload, plotName, filenameBase, plotsKey = 'plots' }) => {
            const data = await postJson(url, { ...payload, format: 'emf' });
            const file = data?.[plotsKey]?.[plotName];
            if (!file || !String(file).startsWith('data:image/emf')) {
                throw new Error(`EMF export unavailable for ${plotName}.`);
            }
            triggerDownload(file, `${slugify(filenameBase || plotName)}.emf`);
        };

        const buildEdaPayload = (format = edaFormat.value, dpi = edaDpi.value) => ({
            col_x: edaColX.value,
            col_y: edaColY.value,
            plot_type: edaPlotType.value,
            dpi,
            format
        });

        const buildCurrentEdaDataPayload = () => ({
            col_x: edaColX.value,
            col_y: edaColY.value,
            plot_type: edaPlotType.value
        });

        const buildTrainPayload = (format = trainFormat.value, dpi = trainDpi.value) => ({
            models: selectedModels.value,
            use_meta: trainUseMeta.value,
            meta_algo: trainMetaAlgo.value,
            meta_algos: trainUseMeta.value && trainMetaCompareAll.value ? trainMetaAlgorithms : [trainMetaAlgo.value],
            ensemble: trainEnsemble.value,
            test_size: trainTestSize.value / 100.0,
            dpi,
            format
        });

        const buildModelPlotPayload = (modelName = plotSelectedModel.value, format = trainFormat.value, dpi = trainDpi.value) => ({
            model_name: modelName,
            interval_bins: Number(trainIntervalBins.value),
            interval_edges: trainIntervalEdges.value,
            dpi,
            format
        });

        const buildInterpPayload = (format = interpFormat.value, dpi = interpDpi.value) => ({ format, dpi });

        const buildPdpPayload = (format = interpFormat.value, dpi = interpDpi.value) => ({
            feature: pdpFeature.value,
            feature2: pdpFeature2.value,
            feature3: pdpFeature3.value,
            format,
            dpi
        });

        const buildCausalPayload = (format = interpFormat.value, dpi = interpDpi.value) => ({
            treatment: causalTreatment.value || null,
            format,
            dpi
        });
        const buildCounterfactualPayload = (format = interpFormat.value, dpi = interpDpi.value) => ({
            sample_idx: counterfactualSampleIdx.value,
            desired_value: counterfactualDesiredValue.value === '' ? null : Number(counterfactualDesiredValue.value),
            format,
            dpi
        });

        const requestEdaImage = (format = edaFormat.value, dpi = edaDpi.value) => postJson('/api/eda', buildEdaPayload(format, dpi));
        const requestAudit = (format = edaFormat.value, dpi = edaDpi.value) => postJson('/api/audit/run', { format, dpi });
        const requestOutliersPlot = (format = prepFormat.value, dpi = prepDpi.value) => postJson('/api/preprocess/plot', { plot_key: 'outliers', format, dpi });
        const requestScalePlot = (format = prepFormat.value, dpi = prepDpi.value) => postJson('/api/preprocess/plot', { plot_key: 'scale', format, dpi });
        const requestAugmentPlot = (format = prepFormat.value, dpi = prepDpi.value) => postJson('/api/preprocess/plot', { plot_key: 'augment', format, dpi });
        const requestAugmentDiagnostics = (format = prepFormat.value, dpi = prepDpi.value) => postJson('/api/preprocess/augment_diagnostics', { format, dpi });
        const requestFeaturePlot = (format = prepFormat.value, dpi = prepDpi.value) => postJson('/api/preprocess/plot', { plot_key: 'features', format, dpi });
        const buildGlobalPlotPayload = (format = trainFormat.value, dpi = trainDpi.value) => ({
            dpi,
            format,
            filter_enabled: trainGlobalFilterEnabled.value,
            min_score: Number(trainGlobalMinR2.value)
        });

        const requestGlobalPlots = (format = trainFormat.value, dpi = trainDpi.value) => postJson('/api/plot_global', buildGlobalPlotPayload(format, dpi));
        const requestModelPlots = (modelName = plotSelectedModel.value, format = trainFormat.value, dpi = trainDpi.value) => postJson('/api/plot_model', buildModelPlotPayload(modelName, format, dpi));
        const requestShapPlots = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/shap', { ...buildInterpPayload(format, dpi), sample_idx: shapSampleIdx.value });
        const requestAdvancedXaiPlots = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/advanced_xai', buildInterpPayload(format, dpi));
        const requestPdpPlots = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/pdp', buildPdpPayload(format, dpi));
        const requestPermutationPlots = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/permutation', buildInterpPayload(format, dpi));
        const requestCausalResult = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/causal', buildCausalPayload(format, dpi));
        const requestCounterfactualResult = (format = interpFormat.value, dpi = interpDpi.value) => postJson('/api/interpret/counterfactual', buildCounterfactualPayload(format, dpi));
        const requestOptimizationSchema = async () => {
            const res = await axios.get('/api/optimization/schema');
            if (res.data.status !== 'success') throw new Error(res.data.message || 'Failed to load optimization schema.');
            return res.data.data;
        };
        const requestFigureCompose = (payload) => postJson('/api/figure/compose', payload);
        const requestModelPackageCatalog = async () => {
            try {
                const res = await axios.get('/api/model_packages');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Failed to load model packages.');
                return res.data;
            } catch (e) {
                return { trained: [], imported: [] };
            }
        };

        const getEdaPlotName = () => {
            const parts = ['EDA', edaPlotType.value || 'plot'];
            if (edaColX.value) parts.push(edaColX.value);
            if (['scatter', 'reg', 'hex', 'joint'].includes(edaPlotType.value) && edaColY.value) parts.push('vs', edaColY.value);
            return parts.join('_');
        };

        const rememberEdaPlot = (dataUrl) => {
            if (!dataUrl) return;
            const name = getEdaPlotName();
            const nextItems = edaPlotItems.value.filter((item) => item.name !== name);
            nextItems.push({ name, data_url: dataUrl });
            edaPlotItems.value = nextItems;
        };

        const handleFileUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await axios.post('/api/upload', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
                if (response.data.status === 'success') {
                    fileData.value = response.data.data;
                    edaImage.value = null;
                    edaPlotItems.value = [];
                    auditResult.value = null;
                    auditPanelOpen.value = false;
                    prepOutlierPlot.value = '';
                    prepScalePlot.value = '';
                    prepAugmentPlot.value = '';
                    prepAugmentDiagnostics.value = {};
                    prepFeaturePlot.value = '';
                    featureImportances.value = [];
                    trainResults.value = [];
                    trainPlots.value = {};
                    globalPlots.value = {};
                    publicationLeakage.value = null;
                    publicationAblation.value = null;
                    publicationExternal.value = null;
                    publicationText.value = '';
                    publicationStoryboard.value = [];
                    publicationAdvanced.value = null;
                    publicationReadiness.value = null;
                    publicationReviewer.value = null;
                    publicationRobustness.value = null;
                    publicationFigureBuilder.value = null;
                    shapPlots.value = {};
                    advancedXaiPlots.value = {};
                    pdpPlots.value = {};
                    permutationPlots.value = {};
                    causalResult.value = null;
                    optimizationSchema.value = null;
                    optimizationResult.value = null;
                    selectedTarget.value = fileData.value.columns[fileData.value.columns.length - 1];
                    edaColX.value = fileData.value.columns[0];
                    await setTarget();
                    await hydrateOptimizationSchema();
                }
            } catch (e) {
                alert('Upload failed: ' + (e.response?.data?.message || e.message));
            }
        };

        const setTarget = async () => {
            try {
                const res = await axios.post('/api/set_target', { target_col: selectedTarget.value });
                problemType.value = res.data.problem_type;
                fetchModels();
            } catch (e) {
                console.error(e);
            }
        };

        const overrideProblemType = async () => {
            try {
                const res = await axios.post('/api/set_problem_type', { problem_type: problemType.value });
                problemType.value = res.data.problem_type;
                fetchModels();
            } catch (e) {
                console.error(e);
            }
        };

        const fetchModels = async () => {
            try {
                const res = await axios.get('/api/models');
                if (res.data.status === 'success') {
                    availableModels.value = res.data.models;
                    if (res.data.models.length > 0) {
                        selectedModels.value = ['Random Forest', 'Logistic Regression', 'XGBoost', 'LightGBM', 'Linear Regression'].filter((m) => res.data.models.includes(m));
                    }
                }
            } catch (e) {}
        };

        const selectAllModels = () => {
            selectedModels.value = [...availableModels.value];
        };

        const clearSelectedModels = () => {
            selectedModels.value = [];
        };

        const hydrateOptimizationSchema = async () => {
            try {
                const schema = await requestOptimizationSchema();
                const catalog = await requestModelPackageCatalog();
                optimizationSchema.value = schema;
                modelPackageCatalog.value = { trained: catalog.trained || [], imported: catalog.imported || [] };
                optAlgorithm.value = schema.algorithms[0] || 'NSGA-II';
                optCompareAlgorithms.value = (schema.compare_default || schema.algorithms || []).slice(0, 4);
                optModelName.value = schema.models[0] || '';
                optShrinkageModelName.value = (catalog.imported || []).find((name) => /shrink|收缩|shrinkage/i.test(name)) || '';
                optVariables.value = (schema.variables || []).map((item) => ({ ...item }));
                optSurrogates.value = [
                    { source: 'imported_model', name: 'Strength', target_col: 'CompressiveStrength', model_name: 'Random Forest', imported_model_name: '', alias: 'strength_pred', goal: 'max', lower: '', upper: '' },
                    { source: 'imported_model', name: 'Flowability', target_col: 'Flowability', model_name: 'Extra Trees', imported_model_name: '', alias: 'flow_pred', goal: 'range', lower: 180, upper: 230 }
                ];
                const v1 = schema.variables?.[0];
                const v2 = schema.variables?.[1];
                optFormulaObjectives.value = [
                    { name: 'Carbon', expression: v1 && v2 ? `${v1.name} * 0.9 + ${v2.name} * 0.12` : 'cement * 0.9 + flyash * 0.12', goal: 'min', lower: '', upper: '' },
                    { name: 'Cost', expression: v1 && v2 ? `${v1.name} * 0.55 + ${v2.name} * 0.2` : 'cement * 0.55 + flyash * 0.2', goal: 'min', lower: '', upper: '' }
                ];
                optConstraintsText.value = 'strength_pred >= 35\nflow_pred >= 180';
            } catch (e) {
                optimizationSchema.value = null;
            }
        };

        const runTraining = async () => {
            if (selectedModels.value.length === 0) return alert('Select at least one layout geometry.');
            trainLoading.value = true;
            try {
                const res = await postJson('/api/train', buildTrainPayload());
                trainResults.value = res.results;
                trainFailures.value = res.failed_models || [];
                plotSelectedModel.value = res.best;
                trainPlots.value = res.plots || {};
                advancedXaiPlots.value = {};
                if (Object.keys(trainPlots.value).length === 0 && res.best) {
                    const modelRes = await requestModelPlots(res.best);
                    trainPlots.value = modelRes.plots || {};
                }
                try {
                    const globalRes = await requestGlobalPlots();
                    globalPlots.value = globalRes.plots || {};
                } catch (e) {
                    globalPlots.value = {};
                }
                await hydrateOptimizationSchema();
                if (trainFailures.value.length > 0) {
                    alert(`训练完成，但有 ${trainFailures.value.length} 个模型失败：\n${trainFailures.value.map((item) => `- ${item.Model}`).join('\n')}`);
                }
            } catch (e) {
                trainFailures.value = [];
                alert(e.response?.data?.message || e.message);
            } finally {
                trainLoading.value = false;
            }
        };

        const refreshTrainingPlots = async () => {
            if (!plotSelectedModel.value) return alert('请先训练模型。');
            trainLoading.value = true;
            try {
                const [globalRes, modelRes] = await Promise.all([
                    requestGlobalPlots(),
                    requestModelPlots(plotSelectedModel.value)
                ]);
                globalPlots.value = globalRes.plots || {};
                trainPlots.value = modelRes.plots || {};
            } catch (e) {
                alert(e.message);
            } finally {
                trainLoading.value = false;
            }
        };

        const generateModelPlots = async (modelName) => {
            if (!modelName) return;
            trainLoading.value = true;
            try {
                const res = await requestModelPlots(modelName);
                trainPlots.value = res.plots || {};
            } catch (e) {
                alert('Plot Generation Failed: ' + e.message);
            } finally {
                trainLoading.value = false;
            }
        };

        const runPublicationLeakageAudit = async () => {
            publicationLoading.value = true;
            try {
                const res = await postJson('/api/publication/leakage_audit', { format: trainFormat.value, dpi: Number(trainDpi.value) });
                publicationLeakage.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runPublicationAblation = async () => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            publicationLoading.value = true;
            try {
                const res = await postJson('/api/publication/ablation', {
                    model_name: plotSelectedModel.value,
                    top_k: Number(publicationAblationTopK.value),
                    format: trainFormat.value,
                    dpi: Number(trainDpi.value)
                });
                publicationAblation.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const triggerPublicationExternalUpload = () => {
            publicationExternalFileRef.value?.click();
        };

        const runPublicationExternalValidation = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            publicationLoading.value = true;
            try {
                const response = await axios.post(`/api/publication/external_validate?model_name=${encodeURIComponent(plotSelectedModel.value || '')}&format=${encodeURIComponent(trainFormat.value)}&dpi=${Number(trainDpi.value)}`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
                if (response.data.status !== 'success') throw new Error(response.data.message || 'External validation failed.');
                publicationExternal.value = response.data.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                event.target.value = '';
                publicationLoading.value = false;
            }
        };

        const generatePublicationText = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/results_text');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Text generation failed.');
                publicationText.value = res.data.data.text;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const loadPublicationStoryboard = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/storyboard');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Storyboard generation failed.');
                publicationStoryboard.value = res.data.data || [];
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const exportReproducibilityPackage = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/reproducibility_package');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Package export failed.');
                downloadBase64File(res.data.content_base64, res.data.filename || 'NatureML_reproducibility_package.zip', 'application/zip');
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runAdvancedScientificDiscovery = async () => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            publicationLoading.value = true;
            try {
                const res = await postJson('/api/publication/advanced_discovery', {
                    model_name: plotSelectedModel.value,
                    top_k: Number(publicationAblationTopK.value) || 12,
                    format: trainFormat.value,
                    dpi: Number(trainDpi.value)
                });
                publicationAdvanced.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runNCReadinessScore = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/readiness_score');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Readiness scoring failed.');
                publicationReadiness.value = res.data.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runReviewerSimulator = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/reviewer_simulator');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Reviewer simulation failed.');
                publicationReviewer.value = res.data.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runRobustnessBattery = async () => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            publicationLoading.value = true;
            try {
                const res = await postJson('/api/publication/robustness_battery', {
                    model_name: plotSelectedModel.value,
                    repeats: Number(publicationRobustnessRepeats.value) || 8,
                    format: trainFormat.value,
                    dpi: Number(trainDpi.value)
                });
                publicationRobustness.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const runFigureManuscriptBuilder = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/figure_manuscript_builder');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Figure builder failed.');
                publicationFigureBuilder.value = res.data.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const exportSubmissionSupportPackage = async () => {
            publicationLoading.value = true;
            try {
                const res = await axios.get('/api/publication/submission_package');
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Submission package export failed.');
                downloadBase64File(res.data.content_base64, res.data.filename || 'NatureML_submission_support_package.zip', 'application/zip');
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                publicationLoading.value = false;
            }
        };

        const generateEDA = async () => {
            if (['scatter', 'reg', 'hex', 'joint'].includes(edaPlotType.value) && !edaColY.value) {
                alert('This plot type requires Feature Y selected.');
                return;
            }
            edaLoading.value = true;
            try {
                const res = await requestEdaImage();
                edaImage.value = res.image;
                rememberEdaPlot(res.image);
            } catch (e) {
                alert('Server generation logic failed: ' + (e.response?.data?.message || e.message));
            } finally {
                edaLoading.value = false;
            }
        };

        const runDataAudit = async () => {
            if (!fileData.value) return alert('请先上传数据集。');
            auditLoading.value = true;
            try {
                const res = await requestAudit();
                auditResult.value = res.data || null;
                auditPanelOpen.value = true;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                auditLoading.value = false;
            }
        };

        const exportEdaSvg = async () => {
            try {
                await downloadSvgFromImageEndpoint({
                    url: '/api/eda',
                    payload: buildEdaPayload('svg', edaDpi.value),
                    filenameBase: `EDA_${edaPlotType.value}_${edaColX.value || 'plot'}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportEdaPdf = async () => {
            try {
                await downloadPdfFromImageEndpoint({
                    url: '/api/eda',
                    payload: buildEdaPayload('pdf', edaDpi.value),
                    filenameBase: `EDA_${edaPlotType.value}_${edaColX.value || 'plot'}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportEdaEmf = async () => {
            try {
                await downloadEmfFromImageEndpoint({
                    url: '/api/eda',
                    payload: buildEdaPayload('emf', edaDpi.value),
                    filenameBase: `EDA_${edaPlotType.value}_${edaColX.value || 'plot'}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const runOutliers = async () => {
            try {
                const res = await postJson('/api/preprocess/outliers', {
                    method: prepOutlierMethod.value,
                    contamination: 0.05,
                    format: prepFormat.value,
                    dpi: prepDpi.value
                });
                alert(`Removed ${res.removed} anomaly vectors. Dimensions updated.`);
                fileData.value.rows = res.rows;
                prepOutlierPlot.value = res.plot || '';
            } catch (e) {
                alert(e.message || e);
            }
        };

        const refreshOutlierPlot = async () => {
            try {
                const res = await requestOutliersPlot();
                prepOutlierPlot.value = res.plot || '';
            } catch (e) {
                alert(e.message);
            }
        };

        const exportOutlierSvg = async () => {
            try {
                const res = await requestOutliersPlot('svg', prepDpi.value);
                triggerDownload(res.plot, 'Outlier_Plot.svg');
            } catch (e) {
                alert(e.message);
            }
        };

        const runScaling = async () => {
            try {
                const res = await postJson('/api/preprocess/scale', {
                    method: prepScaler.value,
                    format: prepFormat.value,
                    dpi: prepDpi.value
                });
                alert('Linear scaling mapping complete.');
                prepScalePlot.value = res.plot || '';
            } catch (e) {
                alert(e.message || e);
            }
        };

        const refreshScalePlot = async () => {
            try {
                const res = await requestScalePlot();
                prepScalePlot.value = res.plot || '';
            } catch (e) {
                alert(e.message);
            }
        };

        const exportScaleSvg = async () => {
            try {
                const res = await requestScalePlot('svg', prepDpi.value);
                triggerDownload(res.plot, 'Scaling_Plot.svg');
            } catch (e) {
                alert(e.message);
            }
        };

        const runAugment = async () => {
            try {
                const res = await postJson('/api/preprocess/augment', {
                    method: prepAugment.value,
                    format: prepFormat.value,
                    dpi: prepDpi.value
                });
                alert(`Data synthesis finished! Row augmentation expanded to: ${res.rows}`);
                fileData.value.rows = res.rows;
                prepAugmentPlot.value = res.plot || '';
                prepAugmentDiagnostics.value = {};
            } catch (e) {
                alert(e.message || e);
            }
        };

        const refreshAugmentPlot = async () => {
            try {
                const res = await requestAugmentPlot();
                prepAugmentPlot.value = res.plot || '';
            } catch (e) {
                alert(e.message);
            }
        };

        const generateAugmentDiagnostics = async () => {
            prepAugmentDiagnosticsLoading.value = true;
            try {
                const res = await requestAugmentDiagnostics();
                prepAugmentDiagnostics.value = res.plots || {};
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                prepAugmentDiagnosticsLoading.value = false;
            }
        };

        const exportAugmentSvg = async () => {
            try {
                const res = await requestAugmentPlot('svg', prepDpi.value);
                triggerDownload(res.plot, 'Augment_Plot.svg');
            } catch (e) {
                alert(e.message);
            }
        };

        const runFeatures = async () => {
            try {
                const res = await postJson('/api/preprocess/features', {
                    method: prepFeatureMethod.value,
                    top_k: prepTopK.value,
                    format: prepFormat.value,
                    dpi: prepDpi.value
                });
                featureImportances.value = res.importances;
                prepFeaturePlot.value = res.plot || '';
            } catch (e) {
                alert(e.message || e);
            }
        };

        const refreshFeaturePlot = async () => {
            try {
                const res = await requestFeaturePlot();
                prepFeaturePlot.value = res.plot || '';
            } catch (e) {
                alert(e.message);
            }
        };

        const exportFeatureSvg = async () => {
            try {
                const res = await requestFeaturePlot('svg', prepDpi.value);
                triggerDownload(res.plot, 'Feature_Plot.svg');
            } catch (e) {
                alert(e.message);
            }
        };

        const generateSHAP = async () => {
            shapLoading.value = true;
            shapPlots.value = {};
            try {
                const res = await requestShapPlots();
                shapPlots.value = res.plots || {};
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                shapLoading.value = false;
            }
        };

        const generateAdvancedXAI = async () => {
            advancedXaiLoading.value = true;
            advancedXaiPlots.value = {};
            try {
                const res = await requestAdvancedXaiPlots();
                advancedXaiPlots.value = res.plots || {};
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                advancedXaiLoading.value = false;
            }
        };

        const exportShapSvg = async (plotName) => {
            try {
                await downloadSvgFromPlotsEndpoint({
                    url: '/api/interpret/shap',
                    payload: { ...buildInterpPayload('svg', interpDpi.value), sample_idx: shapSampleIdx.value },
                    plotName,
                    filenameBase: `SHAP_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const generatePDP = async () => {
            if (!pdpFeature.value) return alert('Select a feature for PDP mapping');
            pdpLoading.value = true;
            pdpPlots.value = {};
            try {
                const res = await requestPdpPlots();
                pdpPlots.value = res.plots || {};
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                pdpLoading.value = false;
            }
        };

        const exportPdpSvg = async (plotName) => {
            try {
                await downloadSvgFromPlotsEndpoint({
                    url: '/api/interpret/pdp',
                    payload: buildPdpPayload('svg', interpDpi.value),
                    plotName,
                    filenameBase: `PDP_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const generateCausal = async () => {
            causalLoading.value = true;
            try {
                const res = await requestCausalResult();
                causalResult.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                causalLoading.value = false;
            }
        };

        const generateCounterfactual = async () => {
            counterfactualLoading.value = true;
            counterfactualResult.value = null;
            try {
                const res = await requestCounterfactualResult();
                counterfactualResult.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                counterfactualLoading.value = false;
            }
        };

        const exportModelPackage = async (modelName) => {
            try {
                const data = await postJson('/api/models/export', { model_name: modelName });
                const binary = atob(data.content_base64);
                const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
                const blob = new Blob([bytes], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = data.filename || `${slugify(modelName)}.naturemlmodel`;
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
                URL.revokeObjectURL(url);
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const exportOptimizationPackage = async () => {
            try {
                const data = await postJson('/api/optimization/export', {});
                const binary = atob(data.content_base64);
                const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
                const blob = new Blob([bytes], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                const anchor = document.createElement('a');
                anchor.href = url;
                anchor.download = data.filename || 'NatureML_Optimization.naturemlopt';
                document.body.appendChild(anchor);
                anchor.click();
                document.body.removeChild(anchor);
                URL.revokeObjectURL(url);
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const importModelPackage = async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await axios.post('/api/models/import', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
                if (res.data.status !== 'success') throw new Error(res.data.message || 'Import failed.');
                await hydrateOptimizationSchema();
                const importedName = res.data.model_name;
                const firstEmpty = optSurrogates.value.find((item) => item.source === 'imported_model' && !item.imported_model_name);
                if (firstEmpty) {
                    firstEmpty.imported_model_name = importedName;
                    if (res.data.target_col && !firstEmpty.target_col) firstEmpty.target_col = res.data.target_col;
                }
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                event.target.value = '';
            }
        };

        const triggerModelImportPicker = () => {
            modelPackageUploadRef.value?.click();
        };

        const triggerFigureUpload = () => {
            figureUploadRef.value?.click();
        };

        const relabelFigurePanels = () => {
            figurePanels.value = figurePanels.value.map((panel, idx) => ({
                ...panel,
                label: panel.label || String.fromCharCode(97 + (idx % 26))
            }));
        };

        const importFigurePanels = async (event) => {
            const files = Array.from(event.target.files || []);
            const readers = files.map((file, idx) => new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve({
                    id: `${Date.now()}_${idx}_${file.name}`,
                    name: file.name,
                    label: String.fromCharCode(97 + ((figurePanels.value.length + idx) % 26)),
                    data_url: reader.result,
                    show_label: true
                });
                reader.onerror = reject;
                reader.readAsDataURL(file);
            }));
            try {
                const nextPanels = await Promise.all(readers);
                figurePanels.value = [...figurePanels.value, ...nextPanels];
                relabelFigurePanels();
            } catch (e) {
                alert('图像导入失败: ' + e.message);
            } finally {
                event.target.value = '';
            }
        };

        const moveFigurePanel = (idx, direction) => {
            const next = [...figurePanels.value];
            const swap = idx + direction;
            if (swap < 0 || swap >= next.length) return;
            [next[idx], next[swap]] = [next[swap], next[idx]];
            figurePanels.value = next;
            relabelFigurePanels();
        };

        const removeFigurePanel = (idx) => {
            figurePanels.value = figurePanels.value.filter((_, panelIdx) => panelIdx !== idx);
            relabelFigurePanels();
        };

        const buildFigurePayload = (formatOverride = 'png') => ({
            panels: figurePanels.value.map((panel) => ({
                name: panel.name,
                label: panel.label,
                data_url: panel.data_url,
                show_label: panel.show_label
            })),
            preset: figurePreset.value,
            columns: Number(figureColumns.value),
            canvas_width: Number(figureCanvasWidth.value),
            canvas_height: Number(figureCanvasHeight.value),
            label_size: Number(figureLabelSize.value),
            label_style: figureLabelStyle.value,
            gap: Number(figureGap.value),
            margin: Number(figureMargin.value),
            format: formatOverride,
            dpi: Number(figureDpi.value)
        });

        const composeFigure = async () => {
            if (!figurePanels.value.length) return alert('请先上传至少一张图。');
            figureLoading.value = true;
            try {
                const res = await requestFigureCompose(buildFigurePayload('png'));
                figurePreview.value = res.image;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                figureLoading.value = false;
            }
        };

        const exportFigureComposition = async () => {
            if (!figurePanels.value.length) return alert('请先上传至少一张图。');
            figureLoading.value = true;
            try {
                const res = await requestFigureCompose(buildFigurePayload(figureFormat.value));
                downloadRenderedPlot(res.image, 'Nature_Figure_Composition', figureFormat.value);
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                figureLoading.value = false;
            }
        };

        const runOptimization = async () => {
            if (!optVariables.value.length) return alert('No optimization variables available.');
            optimizationLoading.value = true;
            optimizationResult.value = null;
            try {
                const surrogateModels = optSurrogates.value.map((item) => ({
                    source: item.source,
                    name: item.name,
                    target_col: item.target_col,
                    model_name: item.model_name,
                    imported_model_name: item.imported_model_name,
                    alias: item.alias,
                }));
                const objectives = [
                    ...optSurrogates.value.map((item) => {
                        const spec = { name: item.name, expression: item.alias, goal: item.goal };
                        if (item.goal === 'range') {
                            spec.lower = Number(item.lower);
                            spec.upper = Number(item.upper);
                        }
                        return spec;
                    }),
                    ...optFormulaObjectives.value.map((item) => {
                        const spec = { name: item.name, expression: item.expression, goal: item.goal };
                        if (item.goal === 'range') {
                            spec.lower = Number(item.lower);
                            spec.upper = Number(item.upper);
                        }
                        return spec;
                    })
                ];
                const constraints = optConstraintsText.value
                    .split(/\r?\n/)
                    .map((line) => line.trim())
                    .filter(Boolean);
                const buildOptimizationPayload = (formatOverride = 'png') => ({
                    model_name: optModelName.value || null,
                    algorithm: optAlgorithm.value,
                    compare_algorithms: optCompareAlgorithms.value,
                    objectives,
                    constraints,
                    surrogate_models: surrogateModels,
                    variables: optVariables.value.map((item) => ({
                        name: item.name,
                        lower: Number(item.lower),
                        upper: Number(item.upper)
                    })),
                    pop_size: Number(optPopulation.value),
                    generations: Number(optGenerations.value),
                    format: formatOverride,
                    dpi: Number(optDpi.value),
                    shrinkage_curve: optShrinkageModelName.value ? {
                        model_name: optShrinkageModelName.value,
                        time_feature: optShrinkageTimeFeature.value || null,
                        time_points: optShrinkageTimePoints.value,
                        top_n: Number(optShrinkageTopN.value) || 5,
                        probabilistic: Boolean(optShrinkageProbabilistic.value)
                    } : null
                });
                const payload = buildOptimizationPayload('png');
                const res = await postJson('/api/optimization/run', payload);
                res.data._request_payload = buildOptimizationPayload;
                optimizationResult.value = res.data;
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                optimizationLoading.value = false;
            }
        };

        const exportOptimizationPlot = async (plotName) => {
            if (!optimizationResult.value?._request_payload) return alert('请先运行优化。');
            try {
                const payload = optimizationResult.value._request_payload(optFormat.value);
                const res = await postJson('/api/optimization/run', payload);
                const image = res?.data?.plots?.[plotName] || res?.data?.comparison_plots?.[plotName];
                if (!image) throw new Error(`未找到图 ${plotName}`);
                downloadRenderedPlot(image, `Optimization_${plotName}`, optFormat.value);
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            }
        };

        const generatePermutation = async () => {
            permutationLoading.value = true;
            permutationPlots.value = {};
            try {
                const res = await requestPermutationPlots();
                permutationPlots.value = res.plots || {};
            } catch (e) {
                alert(e.response?.data?.message || e.message);
            } finally {
                permutationLoading.value = false;
            }
        };

        const exportPermutationSvg = async (plotName) => {
            try {
                await downloadSvgFromPlotsEndpoint({
                    url: '/api/interpret/permutation',
                    payload: buildInterpPayload('svg', interpDpi.value),
                    plotName,
                    filenameBase: `Permutation_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportCausalSvg = async () => {
            try {
                const res = await requestCausalResult('svg', interpDpi.value);
                const graph = res?.data?.graph;
                if (!graph || !String(graph).startsWith('data:image')) {
                    throw new Error('SVG export unavailable for causal graph.');
                }
                triggerDownload(graph, 'Causal_Graph.svg');
            } catch (e) {
                alert(e.message);
            }
        };

        const exportGlobalPlotSvg = async (plotName) => {
            try {
                await downloadSvgFromPlotsEndpoint({
                    url: '/api/plot_global',
                    payload: buildGlobalPlotPayload('svg', trainDpi.value),
                    plotName,
                    filenameBase: plotName
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportGlobalPlotPdf = async (plotName) => {
            try {
                await downloadPdfFromPlotsEndpoint({
                    url: '/api/plot_global',
                    payload: buildGlobalPlotPayload('pdf', trainDpi.value),
                    plotName,
                    filenameBase: plotName
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportGlobalPlotEmf = async (plotName) => {
            try {
                await downloadEmfFromPlotsEndpoint({
                    url: '/api/plot_global',
                    payload: buildGlobalPlotPayload('emf', trainDpi.value),
                    plotName,
                    filenameBase: plotName
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportModelPlotSvg = async (plotName) => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            try {
                await downloadSvgFromPlotsEndpoint({
                    url: '/api/plot_model',
                    payload: buildModelPlotPayload(plotSelectedModel.value, 'svg', trainDpi.value),
                    plotName,
                    filenameBase: `${plotSelectedModel.value}_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportModelPlotPdf = async (plotName) => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            try {
                await downloadPdfFromPlotsEndpoint({
                    url: '/api/plot_model',
                    payload: buildModelPlotPayload(plotSelectedModel.value, 'pdf', trainDpi.value),
                    plotName,
                    filenameBase: `${plotSelectedModel.value}_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        const exportModelPlotEmf = async (plotName) => {
            if (!plotSelectedModel.value) return alert('请先选择模型。');
            try {
                await downloadEmfFromPlotsEndpoint({
                    url: '/api/plot_model',
                    payload: buildModelPlotPayload(plotSelectedModel.value, 'emf', trainDpi.value),
                    plotName,
                    filenameBase: `${plotSelectedModel.value}_${plotName}`
                });
            } catch (e) {
                alert(e.message);
            }
        };

        return {
            backendConnected, currentTab, tabs, handleFileUpload, fileData,
            selectedTarget, problemType, setTarget, overrideProblemType,
            edaColX, edaColY, edaPlotType, edaDpi, edaFormat, edaLoading, edaImage, edaPlotItems, auditLoading, auditResult, auditPanelOpen, generateEDA, runDataAudit, exportEdaSvg, exportEdaPdf, exportEdaEmf, exportCurrentEdaZip, downloadRenderedPlot, exportPlotDataExcel, getEdaPlotName, buildCurrentEdaDataPayload, buildGlobalPlotPayload, buildModelPlotPayload, buildPdpPayload, getPlotHelp,
            prepOutlierMethod, prepScaler, prepAugment, prepFeatureMethod, prepTopK, featureImportances,
            prepOutlierPlot, prepScalePlot, prepAugmentPlot, prepAugmentDiagnostics, prepAugmentDiagnosticsLoading, prepFeaturePlot, prepDpi, prepFormat,
            runOutliers, runScaling, runAugment, runFeatures, exportCurrentPreprocessZip,
            refreshOutlierPlot, refreshScalePlot, refreshAugmentPlot, generateAugmentDiagnostics, refreshFeaturePlot,
            exportOutlierSvg, exportScaleSvg, exportAugmentSvg, exportFeatureSvg,
            availableModels, selectedModels, trainTestSize, trainUseMeta, trainMetaAlgo, trainEnsemble,
            trainLoading, trainResults, trainFailures, trainPlots, globalPlots, plotSelectedModel, trainDpi, trainFormat, trainGlobalFilterEnabled, trainGlobalMinR2, trainIntervalBins, trainIntervalEdges, trainMetaCompareAll, trainMetaAlgorithms,
            publicationLoading, publicationLeakage, publicationAblation, publicationExternal, publicationText, publicationStoryboard, publicationAdvanced, publicationReadiness, publicationReviewer, publicationRobustness, publicationFigureBuilder, publicationExternalFileRef, publicationAblationTopK, publicationRobustnessRepeats,
            modelPlotViewMode, selectedModelPlotCategory, modelPlotCategories, modelPlotCategoryFilters, categorizedModelPlots, visibleCategorizedModelPlots, getModelPlotCategory,
            categoryOverviewWidth, categoryOverviewHeight, categoryOverviewColumns, categoryOverviewDpi, categoryOverviewFormat, exportModelPlotCategoryOverview,
            fetchModels, selectAllModels, clearSelectedModels, runTraining, generateModelPlots, refreshTrainingPlots, runPublicationLeakageAudit, runPublicationAblation, triggerPublicationExternalUpload, runPublicationExternalValidation, generatePublicationText, loadPublicationStoryboard, exportReproducibilityPackage, runAdvancedScientificDiscovery, runNCReadinessScore, runReviewerSimulator, runRobustnessBattery, runFigureManuscriptBuilder, exportSubmissionSupportPackage, getPublicationPlotHelp, exportGlobalPlotSvg, exportGlobalPlotPdf, exportGlobalPlotEmf, exportModelPlotSvg, exportModelPlotPdf, exportModelPlotEmf, exportModelPackage, exportCurrentModelingZip, modelPackageCatalog, modelPackageUploadRef, importModelPackage, triggerModelImportPicker,
            shapLoading, shapPlots, advancedXaiLoading, advancedXaiPlots, shapSampleIdx, pdpFeature, pdpFeature2, pdpFeature3, pdpLoading, pdpPlots,
            permutationLoading, permutationPlots,
            counterfactualSampleIdx, counterfactualDesiredValue, counterfactualLoading, counterfactualResult,
            causalTreatment, causalLoading, causalResult, interpDpi, interpFormat,
            optimizationSchema, optModelName, optAlgorithm, optCompareAlgorithms, optPopulation, optGenerations, optFormat, optDpi, optConstraintsText, optVariables, optimizationLoading, optimizationResult,
            optShrinkageModelName, optShrinkageTimeFeature, optShrinkageTimePoints, optShrinkageTopN, optShrinkageProbabilistic,
            optSurrogates, optFormulaObjectives,
            generateSHAP, generateAdvancedXAI, generatePDP, generatePermutation, generateCounterfactual, generateCausal, getGenerationPlotHelp, exportShapSvg, exportPdpSvg, exportPermutationSvg, exportCausalSvg, exportCurrentInterpretationZip, runOptimization, exportOptimizationPlot, exportOptimizationPackage, exportCurrentOptimizationZip,
            figureUploadRef, figurePanels, figurePreset, figureColumns, figureCanvasWidth, figureCanvasHeight, figureLabelSize, figureLabelStyle, figureGap, figureMargin, figureFormat, figureDpi, figureLoading, figurePreview,
            triggerFigureUpload, importFigurePanels, moveFigurePanel, removeFigurePanel, composeFigure, exportFigureComposition, exportCurrentFigureZip
        };
    }
}).mount('#app');
