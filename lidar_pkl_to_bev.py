import numpy as np
import cv2, pickle,os

#Creates an birds eye view representation of the point cloud data for .pkl.

# Front side (of vehicle) Point Cloud boundary for BEV
# across x axis 0m ~ 26m
# across y axis -10m ~ 10m
# across z axis -0.5m ~ 1.5m
boundary = {
    "minX": 0,
    "maxX": 26,
    "minY": -10,
    "maxY": 10,
    "minZ": -0.5,
    "maxZ": 1.5 #3.0
}

res = 0.1# Desired resolution in metres to use. Each output pixel will represent an square region res x res in size.
BEV_WIDTH = int((boundary["maxY"] - boundary["minY"]) / res)
BEV_HEIGHT = int((boundary["maxX"] - boundary["minX"]) / res)
print('BEV_HEIGHT,BEV_WIDTH:', BEV_HEIGHT,BEV_WIDTH)
DISCRETIZATION = (boundary["maxX"] - boundary["minX"]) / BEV_HEIGHT


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]
    return PointCloud


def makeBVFeature(PointCloud_, Discretization, bc):
    Height = BEV_HEIGHT + 1
    Width = BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)
    PointCloud[:, 2] = PointCloud[:, 2] + abs(bc['minZ'])
    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]
    # Height Map
    heightMap = np.zeros((Height, Width))
    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    # intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    # intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((Height - 1, Width - 1, 3))
    RGB_Map[:, :, 2] = densityMap[:BEV_HEIGHT, :BEV_WIDTH]  # r_map
    RGB_Map[:, :, 1] = heightMap[:BEV_HEIGHT, :BEV_WIDTH]  # g_map
    # RGB_Map[:, :, 0] = intensityMap[:BEV_HEIGHT, :BEV_WIDTH]  # b_map
    return RGB_Map


def main():
    prefix_list = ['up']
    for prefix in prefix_list:
        for i in range(0, 11):  # 2021-08-05-11-38-04(0,4091)
            iterator = "%06d" % (i)
            try:
                dir = './datasets/kitti/velodyne/' + iterator + '.pkl'
                print(dir)
                f = open(dir, 'rb')
                data_dict = pickle.loads(f.read())
                for name, img in data_dict['image'].items():
                    data_dict['image'][name] = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img_cam = data_dict['image'][name] # data_dict['image']['0']

                lidar = data_dict['points']['0-Ouster-OS1-128']
                # pointcloud0 = data_dict['points']['0-Ouster-OS1-128']
                # pointcloud1 =data_dict['points']['2-VLP-16']
                # pointcloud2 = data_dict['points']['1-R-Fans-16']
                # lidar=np.r_[pointcloud0,pointcloud1,pointcloud2]#fuse 3 lidar data
                lidar = removePoints(lidar, boundary)
                img_bev = makeBVFeature(lidar, DISCRETIZATION, boundary)
                img_bev = np.uint8(img_bev * 255)
                img_bev = cv2.rotate(img_bev, cv2.ROTATE_180)


                img_bev_path='./datasets/kitti/velodyne/img_bev/'#save path
                img_cam_path ='./datasets/kitti/velodyne/img_cam/'
                if not os.path.exists(img_bev_path):
                    os.makedirs(img_bev_path)
                if not os.path.exists(img_cam_path):
                    os.makedirs(img_cam_path)

                cv2.imwrite(img_bev_path+ prefix + '_' + iterator + '.png',img_bev)
                cv2.imwrite(img_cam_path + prefix + '_' + iterator + '.png',img_cam)

            except:
                pass


if __name__ == '__main__':
    main()






