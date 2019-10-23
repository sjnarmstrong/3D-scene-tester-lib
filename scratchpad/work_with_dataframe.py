import pandas as pd
import numpy as np
from math import ceil

reconst_res__all_results = pd.read_pickle("test_dta/reconst_res.all_results.pkl")
seg2d_res_class__all_results = pd.read_pickle("test_dta/seg2d_res_class.all_results.pkl")
seg2d_res_class__per_class_results = pd.read_pickle("test_dta/seg2d_res_class.per_class_results.pkl")
seg2d_res_inst__all_results = pd.read_pickle("test_dta/seg2d_res_inst.all_results.pkl")
seg2d_res_seg__all_results = pd.read_pickle("test_dta/seg2d_res_seg.all_results.pkl")
seg3d_res_class__all_results = pd.read_pickle("test_dta/seg3d_res_class.all_results.pkl")
seg3d_res_class__per_class_results = pd.read_pickle("test_dta/seg3d_res_class.per_class_results.pkl")
seg3d_res_inst__all_results = pd.read_pickle("test_dta/seg3d_res_inst.all_results.pkl")
seg3d_res_seg__all_results = pd.read_pickle("test_dta/seg3d_res_seg.all_results.pkl")

name_map = { 'MinkUNet34C_0.02': 'MinkUNet34C (2cm)', 'MinkUNet34C_0.05': 'MinkUNet34C (5cm)',
             'SemanticFusion':'SemanticFusion', '3DMV': '3DMV'
}

alg_names = [""]
# per algorithm results
#for name in name_map.keys():
table = seg3d_res_class__per_class_results.pivot_table(values="iou",
                                                       index=["alg_name"],
                                                       columns=["class"],
                                                       aggfunc=np.nanmean)

table = table.rename(name_map).replace(0, np.NaN)
per_class_iou_tex = table.iloc[:, :table.shape[1]-1].T.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x)\
    .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm")
# print(per_class_iou_tex)


table = seg3d_res_class__per_class_results.pivot_table(values=["inst_acc_num", "inst_acc_den"],
                                                       index=["alg_name"],
                                                       columns=["class"],
                                                       aggfunc=np.sum)
table = 100*table["inst_acc_num"]/table["inst_acc_den"]
table = table.rename(name_map).replace(0, np.NaN)
per_class_acc_tex = table.iloc[:, :table.shape[1]-1].T.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x)\
    .replace('nan', '-').replace("class ", "Class ").replace("alg\\_name", "Algorithm")


table = seg3d_res_class__all_results.pivot_table(values=["miou", "fiou"], index=["alg_name"], aggfunc=np.nanmean)
t2 = seg3d_res_class__all_results.pivot_table(values=["pt_acc_num", "pt_acc_den"], index=["alg_name"], aggfunc=np.nansum)
table.insert(0, 'Accuracy (%)', 100*t2["pt_acc_num"]/t2["pt_acc_den"])

table = table.rename(name_map).replace(0, np.NaN)
overall_accuracy_tex = table.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x)\
    .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm").replace("fiou", "FIoU")\
    .replace("miou", "MIoU")


reconst_res__all_results["rec_var"] = reconst_res__all_results["reconstruction_error_std"] ** 2
table = reconst_res__all_results.pivot_table(values=["reconstruction_error_mean", "reconstruction_error_mse", "rec_var"],
                                     index=["alg_name"], aggfunc=np.nanmean)
t1 = reconst_res__all_results.pivot_table(values=["reconstruction_error_mean"],
                                     index=["alg_name"], aggfunc=np.nanmax)
table["STD. (m)"] = np.sqrt(table["rec_var"])
table["Max Mean Error (m)"] = t1["reconstruction_error_mean"]
table["RMSE (m)"] = np.sqrt(table["reconstruction_error_mse"])
table = table.rename(name_map).replace(0, np.NaN)
reconstruction_res_tex = table.to_latex(columns=["reconstruction_error_mean", "STD. (m)", "Max Mean Error (m)",
                                                 "RMSE (m)"],
                                        na_rep='-',
                                        float_format=lambda x: '%.3f' % x)\
    .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm")\
    .replace("reconstruction\\_error\\_mean", "Mean (m)")


reconst_res__all_results["var"] = reconst_res__all_results["std"] ** 2
reconst_res__all_results["mse"] = reconst_res__all_results["rmse"] ** 2
table = odom_res__all_results.pivot_table(values=["mean", "mse", "var"], index=["alg_name"], aggfunc=np.nanmean)
table['max'] = odom_res__all_results.pivot_table(values=["max"], index=["alg_name"], aggfunc=np.nanmax)['max']
table['min'] = odom_res__all_results.pivot_table(values=["min"], index=["alg_name"], aggfunc=np.nanmin)['min']
table['rmse'] = np.sqrt(table["mse"])
table['std'] = np.sqrt(table["var"])
table = table.rename(name_map).replace(0, np.NaN)
odom_res_tex = table.to_latex(columns=["mean", "std", "min", "max", "rmse"],
                                        na_rep='-',
                                        float_format=lambda x: '%.3f' % x)\
    .replace('nan', '-').replace("class ", "Class ").replace("alg\_name", "Algorithm")\
    .replace("mean", "Mean (m)").replace("std", "STD. (m)").replace("min", "min (m)").replace("max", "max (m)").replace("rmse", "rmse (m)")

print(odom_res_tex)

# print(reconstruction_res_tex)
# print(overall_accuracy_tex)
# print(per_class_acc_tex)


# split = ceil((IoU_Table.shape[1]-1) / 2)
# lres1 = IoU_Table.iloc[:, :split].to_latex(na_rep='-', float_format=lambda x: '%.3f' % x).replace('nan', '-')\
#     .replace('nan', '-').replace("class &  ", "Class &  ").replace("alg\_name", "Algorithm")
# lres2 = IoU_Table.iloc[:, split:IoU_Table.shape[1]-1].to_latex(na_rep='-', float_format=lambda x: '%.3f' % x)\
#     .replace('nan', '-').replace("class &  ", "Class &  ").replace("alg\\_name", "Algorithm")
# print(lres1)
# print(lres2)

#
#
# IoU_Table = seg3d_res_class__per_class_results.pivot_table(values=["inst_acc_num", "inst_acc_den"],
#                                                            index=["alg_name"],
#                                                            columns=["class"],
#                                                            aggfunc=np.nanmean)

# IoU_Table = seg3d_res_class__per_class_results.pivot_table(values="iou",
#                                                            index=["alg_name"],
#                                                            columns=["class"],
#                                                            aggfunc=np.nanmean)
#
# IoU_Table = IoU_Table.rename(name_map).replace(0, np.NaN)
# lres = IoU_Table.to_latex(na_rep='-', float_format=lambda x: '%.3f' % x)