port=10000
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ratios=(0.1)
root_path=datasets/$task/
log_path=logs/$task
for file in $(ls $root_path);
do
    if [[ $file == episode_13 ]];then
    ((port=$RANDOM))
    # if [[ $DEVICEID == 0 ]];then
        # echo "run"
    CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s $root_path/$file \
        -m $log_path/${file} \
        --port $port \
        --pcd-path point_cloud_0.1.pcd \
        --sh-degree 0 --checkpoint_iterations 1 7000 10000 15000 \
        --test_iterations 1 500 1000 2000 4000 8000 15000 \
        --iterations 15000 \
        --feature-file features_0.1.npy &
    # fi
    ((DEVICEID=$DEVICEID+1))
    ((DEVICEID=$DEVICEID%$DEVICENUM))
    fi
    # for ratio in ${ratios[*]};do
    #     ((port=$port+5000))
    #     CUDA_VISIBLE_DEVICES=$DEVICEID python train_depth.py -s ./real_data/$file -m logs/depth/$file_$ratio --port $port --pcd-path point_cloud_$ratio.pcd &
    #     ((DEVICEID=$DEVICEID+1))
    #     ((DEVICEID=$DEVICEID%$DEVICENUM))
    # done
        # echo $file
    # fi
done
wait
echo "training finish"
# python train.py -s ./real_data/scene_0001 --port 12375


