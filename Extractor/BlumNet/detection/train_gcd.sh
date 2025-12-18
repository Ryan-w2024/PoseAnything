# Download pretrained backbones
# bash detection/gcd/download_backbones.sh

# Download datasets sk1491 [GoogleDrive](https://drive.google.com/file/d/11ya3dDYnbiUEAElz9aZVnf6aN5uTg77F/view?usp=sharing)

# start training
export PYTHONPATH=$PYTHONPATH:.
out_pts=128

data_name="sk1491"    # "sk506"       "WH-SYMMAX"    "SymPASCAL"
backbone="swin_base"  # "swin_small"  "vgg16"

work_dir=exps/${backbone}_${data_name}_curves_pts
mkdir -p ${work_dir}

python detection/gcd/train.py --aux_loss --gid --out_pts ${out_pts} --output_dir ${work_dir} --num_feature_levels 3 --backbone ${backbone} --dataset_file ${data_name}  2>&1 | tee ${work_dir}/training.log

