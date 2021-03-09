import os
import argparse
import tensorflow as tf
import keras.backend as K
import pickle

from glob import glob

from lib.io import openpose_from_file, read_segmentation, write_mesh, write_mesh_custom
from model.octopus import Octopus


def main(weights, name, segm_dir, pose_dir, out_dir, opt_pose_steps, opt_shape_steps, opt_size, cal_trans):
    segm_files = sorted(glob(os.path.join(segm_dir, '*.png')))
    pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))

    if len(segm_files) != len(pose_files) or len(segm_files) == len(pose_files) == 0:
        exit('Inconsistent input.')

    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

    model = Octopus(num=len(segm_files))
    #Octopus 모델은 그대로 1080,1080 기준으로 생성
    model.load(weights)

    segmentations = [read_segmentation(f) for f in segm_files]

    joints_2d, face_2d = [], []
    for f in pose_files:
        #pose를 720 기준으로 스케일링
        j, f = openpose_from_file(f, resolution=(opt_size,opt_size))

        assert(len(j) == 25)
        assert(len(f) == 70)

        joints_2d.append(j)
        face_2d.append(f)

    if opt_pose_steps:
        print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, opt_steps=opt_pose_steps)

    if opt_shape_steps:
        print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d, opt_steps=opt_shape_steps)

    print('Estimating shape...')
    pred = model.predict(segmentations, joints_2d)

    os.makedirs(out_dir, exist_ok=True)
    if cal_trans:
        pickle_out = open('{}/octopus_trans.pkl'.format(out_dir), "wb")
        pickle.dump([list(pred['trans'][0])], pickle_out)
        pickle_out.close()
        return
    write_mesh_custom(os.path.join(out_dir,name+'.obj'), pred['vertices'][0], pred['faces'])
    width = 1080
    height = 1080
    camera_c = [540.0, 540.0]
    camera_f = [1080, 1080]
    vertices = pred['vertices']

    data_to_save = {'width': width, 'camera_c': camera_c, 'vertices': vertices, 'camera_f': camera_f, 'height':height }

    pickle_out = open('{}/frame_data.pkl'.format(out_dir), "wb")
    pickle.dump(data_to_save, pickle_out)
    pickle_out.close()

    print('Done.')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dir',
        type=str,
        help="Dataset dir")

    parser.add_argument(
        '--opt_steps_pose', '-p', default=20, type=int,
        help="Optimization steps pose")

    parser.add_argument(
        '--opt_steps_shape', '-s', default=30, type=int,
        help="Optimization steps")

    parser.add_argument(
        '--weights', '-w',
        default='weights/octopus_weights.hdf5',
        help='Model weights file (*.hdf5)')

    parser.add_argument(
        '--size',
        type=int,
        default=1080)

    parser.add_argument(
        '--cal_trans',
        action='store_true')

    args = parser.parse_args()
    name = ' '.join((args.dir).split('/')).split()[-1]
    segm_dir = os.path.join(args.dir,'segmentations')
    pose_dir = os.path.join(args.dir, 'keypoints')
    out_dir = args.dir

    main(args.weights, name, segm_dir, pose_dir, out_dir, args.opt_steps_pose, args.opt_steps_shape, args.size, args.cal_trans)
