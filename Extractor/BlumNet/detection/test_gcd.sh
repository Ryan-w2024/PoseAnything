# eval F1-score
export PYTHONPATH=$PYTHONPATH:.
out_pts=128

data_name="sk1491"    # "sk506"       "WH-SYMMAX"    "SymPASCAL"
backbone="swin_base"  # "swin_small"  "vgg16"

work_dir=exps/${backbone}_${data_name}_curves_pts

python detection/gcd/f1_score.py --aux_loss --gid --out_pts ${out_pts} --resume ${work_dir}/checkpoint.pth  --num_feature_levels 3 --backbone ${backbone} --dataset_file ${data_name}