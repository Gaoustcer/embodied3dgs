dataset_root=datasets/$task
# save_root=datasets/grasp_pikachu
# subpaths=(new_scene single_obj)
for expname in $(ls $dataset_root);do
    # echo $expname
    python feature_pcd.py --root-path $dataset_root/$expname/ --cam2hand datasets/Scene/cam2hand.npy --feature-level 0 &
done
wait
for expname in $(ls $dataset_root);do
    python tools/down_sampling_pcd.py --pcd-root-path $dataset_root/$expname \
        --down-sample-ratio 0.1 &
done
wait
