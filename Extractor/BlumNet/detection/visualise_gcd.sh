# show graph component
export PYTHONPATH=$PYTHONPATH:.
out_pts=128

data_name="sk1491"    # "sk506"       "WH-SYMMAX"    "SymPASCAL"
backbone="swin_base"  # "swin_small"  "vgg16"
visual_type="lines&pts"  # "lines"       "lines&pts"

work_dir=exps/${backbone}_${data_name}_curves_pts

python detection/gcd/test.py --aux_loss --gid --visual_type ${visual_type} --out_pts ${out_pts} \
 --resume ${work_dir}/checkpoint.pth  --num_feature_levels 3 --backbone ${backbone} --dataset_file ${data_name}