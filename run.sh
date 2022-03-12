./debug.sh demo/demo.py     --config-file configs/BoxInst/Base-BoxInst3D.yaml     --input input1.jpg   -o output1.jpg

CUDA_VISIBLE_DEVICES=1 python demo/demo3d.py  --config-file configs/BoxInst/Base-BoxInst3D.yaml  --input /home/hynx/kits21/data/case_00001/imaging.nii.gz

OMP_NUM_THREADS=1  ./debug.sh tools/train_net.py     --config-file configs/BoxInst/Base-BoxInst3D.yaml     --num-gpus 1     OUTPUT_DIR training_dir/BoxInst/Base-BoxInst3D_1

python demo/demo.py     --config-file configs/BoxInst/Base-BoxInst.yaml     --input /home/hynx/kits21/data/case_00001/imaging.nii.gz

OMP_NUM_THREADS=1 ./debug.sh tools/train_net.py     --config-file configs/BoxInst/Base-BoxInst.yaml     --num-gpus 1     OUTPUT_DIR training_dir/BoxInst/Base-BoxInst_tmp

CUDA_VISIBLE_DEVICES=1 python demo/demo.py  --config-file configs/BoxInst/Base-BoxInst.yaml  --input /home/hynx/kits21/data/case_00001/imaging.nii.gz

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python tools/train_net.py     --config-file configs/BoxInst/Base-BoxInst.yaml     --num-gpus 1     OUTPUT_DIR training_dir/BoxInst/Base-BoxInst_pretrain_100