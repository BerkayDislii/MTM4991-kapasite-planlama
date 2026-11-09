// Kaynakça (DD-050 3.8 biçimi). Anahtar → kaynak metni. Numara, metindeki ilk atıf sırasına göre derleme sırasında verilir.
// İkiden fazla yazarlı İngilizce yayınlarda ilk iki yazar + "et al." yazılır.
const ERISIM = "(Erişim tarihi: 02.11.2026)";
module.exports = {
  // A. Bulut ve Kubernetes
  armbrust: 'Armbrust, M. ve Fox, A., et al., (2010), "A View of Cloud Computing", Communications of the ACM, Vol.53, No.4, pp.50-58.',
  burns: 'Burns, B. ve Grant, B., et al., (2016), "Borg, Omega, and Kubernetes", Communications of the ACM, Vol.59, No.5, pp.50-57.',
  reiss: 'Reiss, C. ve Tumanov, A., et al., (2012), "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis", Third ACM Symposium on Cloud Computing (SoCC \'12), San Jose, pp.1-13.',
  rzadca: 'Rzadca, K. ve Findeisen, P., et al., (2020), "Autopilot: Workload Autoscaling at Google", Fifteenth European Conference on Computer Systems (EuroSys \'20), Heraklion, pp.1-16.',
  k8s_kaynak: `The Kubernetes Authors, (2026), "Resource Management for Pods and Containers", https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/, ${ERISIM}.`,
  k8s_hpa: `The Kubernetes Authors, (2026), "Horizontal Pod Autoscaling", https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/, ${ERISIM}.`,
  k8s_dugum: `The Kubernetes Authors, (2026), "Node Autoscaling", https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/, ${ERISIM}.`,
  keda: `KEDA Authors, (2026), "KEDA: Kubernetes Event-driven Autoscaling", https://keda.sh/, ${ERISIM}.`,
  aws_fiyat: 'Amazon Web Services, (2026), "AWS Price List Bulk API: Amazon EC2, eu-central-1", https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/eu-central-1/index.csv, (Erişim tarihi: 29.10.2026).',
  // B. Veri kümeleri
  shen: 'Shen, S. ve van Beek, V., et al., (2015), "Statistical Characterization of Business-Critical Workloads Hosted in Cloud Datacenters", 15th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGrid), Shenzhen, pp.465-474.',
  guo: 'Guo, J. ve Chang, Z., et al., (2019), "Who Limits the Resource Efficiency of My Datacenter: An Analysis of Alibaba Datacenter Traces", International Symposium on Quality of Service (IWQoS \'19), Phoenix, pp.1-10.',
  // C. İstatistiksel tahmin
  calheiros: 'Calheiros, R.N. ve Masoumi, E., et al., (2015), "Workload Prediction Using ARIMA Model and Its Impact on Cloud Applications\' QoS", IEEE Transactions on Cloud Computing, Vol.3, No.4, pp.449-458.',
  roy: 'Roy, N. ve Dubey, A., et al., (2011), "Efficient Autoscaling in the Cloud Using Predictive Models for Workload Forecasting", IEEE 4th International Conference on Cloud Computing (CLOUD), Washington, pp.500-507.',
  box: 'Box, G.E.P. ve Jenkins, G.M., et al., (2015), Time Series Analysis: Forecasting and Control, 5th ed., Wiley, Hoboken.',
  hyndman_kitap: 'Hyndman, R.J. ve Athanasopoulos, G., (2021), Forecasting: Principles and Practice, 3rd ed., OTexts, Melbourne.',
  dickey: 'Dickey, D.A. ve Fuller, W.A., (1979), "Distribution of the Estimators for Autoregressive Time Series with a Unit Root", Journal of the American Statistical Association, Vol.74, No.366, pp.427-431.',
  kpss: 'Kwiatkowski, D. ve Phillips, P.C.B., et al., (1992), "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root", Journal of Econometrics, Vol.54, No.1-3, pp.159-178.',
  akaike: 'Akaike, H., (1974), "A New Look at the Statistical Model Identification", IEEE Transactions on Automatic Control, Vol.19, No.6, pp.716-723.',
  cleveland: 'Cleveland, R.B. ve Cleveland, W.S., et al., (1990), "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", Journal of Official Statistics, Vol.6, No.1, pp.3-73.',
  wang: 'Wang, X. ve Smith, K., et al., (2006), "Characteristic-Based Clustering for Time Series Data", Data Mining and Knowledge Discovery, Vol.13, No.3, pp.335-364.',
  hyndman_koehler: 'Hyndman, R.J. ve Koehler, A.B., (2006), "Another Look at Measures of Forecast Accuracy", International Journal of Forecasting, Vol.22, No.4, pp.679-688.',
  // D. Derin öğrenme
  hochreiter: 'Hochreiter, S. ve Schmidhuber, J., (1997), "Long Short-Term Memory", Neural Computation, Vol.9, No.8, pp.1735-1780.',
  kumar: 'Kumar, J. ve Goomer, R., et al., (2018), "Long Short Term Memory Recurrent Neural Network (LSTM-RNN) Based Workload Forecasting Model for Cloud Datacenters", Procedia Computer Science, Vol.125, pp.676-682.',
  zeng: 'Zeng, A. ve Chen, M., et al., (2023), "Are Transformers Effective for Time Series Forecasting?", Proceedings of the AAAI Conference on Artificial Intelligence, Vol.37, No.9, pp.11121-11128.',
  // E. Tamsayılı programlama
  speitkamp: 'Speitkamp, B. ve Bichler, M., (2010), "A Mathematical Programming Approach for Server Consolidation Problems in Virtualized Data Centers", IEEE Transactions on Services Computing, Vol.3, No.4, pp.266-278.',
  beloglazov: 'Beloglazov, A. ve Buyya, R., (2012), "Optimal Online Deterministic Algorithms and Adaptive Heuristics for Energy and Performance Efficient Dynamic Consolidation of Virtual Machines in Cloud Data Centers", Concurrency and Computation: Practice and Experience, Vol.24, No.13, pp.1397-1420.',
  lin: 'Lin, M. ve Wierman, A., et al., (2013), "Dynamic Right-Sizing for Power-Proportional Data Centers", IEEE/ACM Transactions on Networking, Vol.21, No.5, pp.1378-1391.',
  carrion: 'Carrión, M. ve Arroyo, J.M., (2006), "A Computationally Efficient Mixed-Integer Linear Formulation for the Thermal Unit Commitment Problem", IEEE Transactions on Power Systems, Vol.21, No.3, pp.1371-1378.',
  wolsey: 'Wolsey, L.A., (2020), Integer Programming, 2nd ed., Wiley, Hoboken.',
  achterberg: 'Achterberg, T., (2009), "SCIP: Solving Constraint Integer Programs", Mathematical Programming Computation, Vol.1, No.1, pp.1-41.',
  ortools: `Perron, L. ve Furnon, V., (2026), "OR-Tools", Sürüm 9.15, Google, https://developers.google.com/optimization/, ${ERISIM}.`,
  // F. Otomatik ölçekleme
  lorido: 'Lorido-Botran, T. ve Miguel-Alonso, J., et al., (2014), "A Review of Auto-scaling Techniques for Elastic Applications in Cloud Environments", Journal of Grid Computing, Vol.12, No.4, pp.559-592.',
  qu: 'Qu, C. ve Calheiros, R.N., et al., (2018), "Auto-Scaling Web Applications in Clouds: A Taxonomy and Survey", ACM Computing Surveys, Vol.51, No.4, pp.1-33.',
  urgaonkar: 'Urgaonkar, B. ve Shenoy, P., et al., (2008), "Agile Dynamic Provisioning of Multi-Tier Internet Applications", ACM Transactions on Autonomous and Adaptive Systems, Vol.3, No.1, pp.1-39.',
  gong: 'Gong, Z. ve Gu, X., et al., (2010), "PRESS: PRedictive Elastic ReSource Scaling for Cloud Systems", International Conference on Network and Service Management (CNSM), Niagara Falls, pp.9-16.',
  gandhi: 'Gandhi, A. ve Harchol-Balter, M., et al., (2012), "AutoScale: Dynamic, Robust Capacity Management for Multi-Tier Data Centers", ACM Transactions on Computer Systems, Vol.30, No.4, pp.1-26.',
  toka: 'Toka, L. ve Dobreff, G., et al., (2021), "Machine Learning-Based Scaling Management for Kubernetes Edge Clusters", IEEE Transactions on Network and Service Management, Vol.18, No.1, pp.958-972.',
  patil: 'Patil, S. ve D G, N., et al., (2025), "Dynamic Autoscaling and Scheduling in Kubernetes Clusters with LSTM and ILP", Journal of Communications Software and Systems, Vol.21, No.4, pp.465-476.',
  imdoukh: 'Imdoukh, M. ve Ahmad, I., et al., (2020), "Machine Learning-Based Auto-Scaling for Containerized Applications", Neural Computing and Applications, Vol.32, No.13, pp.9745-9760.',
};
