DEVICE=0
# tasks=(lamp_on open_box reach_target)
# for task in ${tasks[*]};do
rootpath=datasets/$task/
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for episode in $(ls $rootpath);do
    echo $episode
    dataset_path=$rootpath/$episode
    CUDA_VISIBLE_DEVICES=${DEVICE} python preprocess_clip.py --dataset_path $dataset_path &
    # ((DEVICE=$DEVICE+1))
    if [[ $DEVICE == 0 ]]; then
        wait
    fi
    ((DEVICE=$DEVICE+1))
    ((DEVICE=$DEVICE%$DEVICENUM))
done
    # rootpath=datasets/rlbench/gaussian_dataset/$task
    # steps=(step_0 step_1 step_2 final_state)
echo "loop over"
wait
