#!/usr/bin/python3
# -*- coding:UTF-8 -*-
import time
import pyrealsense2 as rs
import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
import platform
import ctypes
from ctypes import *
import os


pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
align_to = rs.stream.color
align = rs.align(align_to)

# mediapipe对应面部的468个关键点
all_list = [x for x in range(0, 477)]
# 额外添加关键点***************************************(李博)
mouthlist = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 308, 415, 310, 311, 312, 13, 82, 80, 191, 78,
             95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,316,86,404,321,302,303,164,200]
             #4, 5,175 ,199, 10, 151,280,352,50,123]
# 4, 5,175 199, 10 151,280,352,,50,123
#**********************

# [array([0.04 , 0.024, 0.222]), array([0.04 , 0.033, 0.218]), array([0.04 , 0.033, 0.218]), array([0.041, 0.042, 0.214]),
#  array([0.033, 0.024, 0.221]), array([0.026, 0.029, 0.218]), array([0.022, 0.033, 0.218]), array([0.018, 0.042, 0.22 ]), 
# array([0.02 , 0.041, 0.22 ]), array([0.026, 0.037, 0.218]), array([0.035, 0.033, 0.218]), array([0.034, 0.042, 0.214]), 
# array([0.035, 0.034, 0.218]), array([0.026, 0.037, 0.218]), array([0.024, 0.043, 0.219]), array([0.022, 0.039, 0.219]), 
# array([0.02 , 0.042, 0.22 ]), array([0.03 , 0.036, 0.218]), array([0.029, 0.043, 0.216]), array([0.019, 0.038, 0.219]), 
# array([0.022, 0.039, 0.219]), array([0.046, 0.023, 0.225]), array([0.053, 0.026, 0.229]), array([0.056, 0.028, 0.227]), 
# array([0.061, 0.036, 0.23 ]), array([0.058, 0.035, 0.228]), array([0.052, 0.033, 0.223]), array([0.049, 0.032, 0.222]), 
# array([0.044, 0.032, 0.22 ]), array([0.047, 0.04 , 0.217]), array([0.045, 0.033, 0.22 ]), array([0.052, 0.033, 0.223]), 
# array([0.058, 0.039, 0.229]), array([0.055, 0.034, 0.225]), array([0.059, 0.037, 0.229]), array([0.049, 0.033, 0.222]), 
# array([0.053, 0.04 , 0.223]), array([0.059, 0.032, 0.229]), array([0.055, 0.034, 0.225])]

mydata=[np.array([1,2,3]),
    np.array([4,5,3]),
    np.array([7,8,3])]
# 计算法向************************************(李博)
# 由于太稀疏，不够稳定，没用上
def compute_pionts_normal(ps):
    sum_x=0
    sum_y=0
    sum_z=0
    for i in range(len(ps)):
        sum_x+= ps[i][0]
        sum_y+=ps[i][1]
        sum_z+=ps[i][2]
    avg=np.array([sum_x/len(ps),sum_y/len(ps),sum_z/len(ps)])
    out=ps -avg
    ##data = mat([[1,2,3],[4,5,6]])
    U,sigma,VT=np.linalg.svd(out)

    # print(VT)

    re=np.array([VT[0][2],VT[1][2],VT[2][2]])
    if re[2]>0:
        re[0]=-re[0]
        re[1]=-re[1]
        re[2]=-re[2]
    print(re)
    # print(out)
    
    return re

def my_require1(ps):
    dllname=""
    if(platform.system()=="Windows"):
        dllname="libpypoints.dll"
    elif(platform.system()=="Windows"):
        dllname="libpypoints.so"
    else:
        dllname="libpypoints.so" 
    lib=ctypes.cdll.LoadLibrary(('{0}/temp_libs/'+dllname).format(os.path.join(os.getcwd())))
   
    pdouble=POINTER(c_double)

    pdata=(pdouble*len(ps))()
 
    for i in range(0,len(ps)):
        pdata[i]=(c_double*3)()
        for j in range(0,3):
            pdata[i][j]=ps[i][j]
    # lib.ctest.restype=ctypes.POINTER(ctypes.c_double*3*len(ps))

    # re=lib.py_compute_stabilize_normal(byref(pdata),len(ps))
    lib.py_compute_stabilize_normal(byref(pdata),len(ps))
    

    return pdata 
# *********************


## numpy 
## u,s,v=np.linalg.svd(M)



class Detection:
    def __init__(self):

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.x_temp1 = []
        self.y_temp1 = []


    def filter(self, point_coordinate, n):
        '''
        均值滤波器，n为滤波强度
        :param point_coordinate: 待滤波的坐标点
        :param n: 均值滤波的强度，平均每n次求一次均值
        :return: 滤波后的坐标点
        '''
        if len(self.x_temp1) < (n + 1):
            self.x_temp1.append(point_coordinate[0])
            self.y_temp1.append(point_coordinate[1])
            point_last = [np.round(np.mean(self.x_temp1), 3), np.round(np.mean(self.y_temp1), 3)]
            return point_last
        else:
            self.x_temp1.append(point_coordinate[0])
            self.y_temp1.append(point_coordinate[1])
            del self.x_temp1[0]
            del self.y_temp1[0]
            point_last = [np.round(np.mean(self.x_temp1), 3), np.round(np.mean(self.y_temp1), 3)]
            return point_last


    def get_depth(self, px, py, depth_data):
        '''
        确保每次像素值到深度值的完全一一映射
        :param px: 像素点横坐标Px
        :param py: 像素点纵坐标Py
        :param depth_data:
        :return:
        '''
        useful_num = 0
        useful_value_total = 0
        calc_pow = 0
        while useful_num < 0.1:
            for i in range((px - calc_pow), (px + calc_pow)):
                for j in range((py - calc_pow), (py + calc_pow)):
                    if i >= 640 or j >= 480 or i <= 0 or j <= 0:
                        continue
                    if depth_data.get_distance(i, j) > 0.01 and depth_data.get_distance(i, j) < 10000:
                        useful_num = useful_num + 1
                        useful_value_total += depth_data.get_distance(i, j)
            calc_pow = calc_pow + 1

        return useful_value_total / useful_num


    def D435(self,frame):
        depth_frame = align.process(frame).get_depth_frame()

        # =============================复合过滤器处理深度图==============================
        depth_to_disparity = rs.disparity_transform(True)                            # 深度表示转换为视差形式
        disparity_to_depth = rs.disparity_transform(False)
        # decimation=rs.decimation_filter()                                          # 降采样
        # decimation.set_option(rs.option.filter_magnitude,4)
        
        spatial = rs.spatial_filter()  
# 添加空间过滤参数******************************************(李博)

        spatial.set_option(rs.option.filter_magnitude,2)
        spatial.set_option(rs.option.filter_smooth_alpha,0.5)
        spatial.set_option(rs.option.filter_smooth_delta,20)
 # 添加时间过滤参数*********************(李博)

                                                      # 空间过滤器
        temporal = rs.temporal_filter()  
        temporal.set_option(rs.option.filter_smooth_alpha,0.5)
        temporal.set_option(rs.option.filter_smooth_delta,20)
 # 添加时间过滤参数*********************

                                 # 时间过滤器
        hole_filling = rs.hole_filling_filter()                                      # 孔填充

        # depth_frame=decimation.process(depth_frame)

        depth_frame = depth_to_disparity.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        depth_frame.__class__ = rs.pyrealsense2.depth_frame
        # ============================================================================

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics       # 获取深度参数（像素坐标系转相机坐标系会用到）

        return depth_frame, depth_intrin


    def cv_show(self, images, rect, pp1, fps, mouth_xyz):

        cv2.circle(images, (int(rect[0][0]), int(rect[0][1])), 4, [0, 255, 255], -1)  # 画出嘴部中心点 黄点
        # cv2.line(images, (mid_x, mid_y), pp2, (0, 0, 255), 2)                       # ==== 绘制嘴部法向量 ======
        cv2.line(images, (int(rect[0][0]), int(rect[0][1])), pp1, (0, 0, 255), 2)     # ==== 绘制嘴部法向量 ======

        cv2.putText(images, "FPS= %.d" % fps, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(images, "X: %.5f m" % mouth_xyz[0], (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(images, "Y: %.5f m" % mouth_xyz[1], (0, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(images, "Z: %.5f m" % mouth_xyz[2], (0, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.namedWindow('MediaPipe Face Mesh', cv2.WINDOW_NORMAL)
        cv2.imshow('MediaPipe Face Mesh', images)

        cv2.waitKey(5)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break


    def run(self):
        try:
            vis = o3d.visualization.Visualizer()
            # 创建播放窗口
            vis.create_window()
            pointcloud = o3d.geometry.PointCloud()
            to_reset = True
            vis.add_geometry(pointcloud)

            while True:
                start_time = time.time()

                frame = pipe.wait_for_frames()
                color_frame = align.process(frame).get_color_frame()
                depth_frame, depth_intrin = self.D435(frame)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # 显示depth图像：伪彩色图操作
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)       # BGR --> RGB
                image.flags.writeable = True

                results = self.face_mesh.process(image)                    # 人脸识别
                if results.multi_face_landmarks:
                    multi_face = results.multi_face_landmarks              # 推理出图像中的人脸landmark，只设定了一个

                    one_face = multi_face[0]                               # 取出图像中的唯一的人脸信息--landmark

                    # ============================== 循环开始 ================================
                    cnts = []                                              # 定位嘴部四个关键点位置
                    mouth_plane = []                                       # 嘴部点云集合面，作提取嘴部法向量用
                    face = []
                    face_plane = []

                    for id, landmark in enumerate(one_face.landmark):
                        image_height, image_width, image_channel = image.shape                 # --> 480, 640, 3
                        x, y = int(landmark.x * image_width), int(landmark.y * image_height)   # --> 对应于图像尺寸的 面部关键点 x, y值

                        if id in mouthlist:
                            ## print(id)
                            cnts.append([x, y])                                                # cnts --> 嘴部的关键点列表
                            diatance = self.get_depth(x, y, depth_frame)
                            ##diatance=round(diatance,2)

                            ##print(diatance)

                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], diatance)
                            #depth_point = np.round(depth_point, 3)
                            ##print(depth_point.type())
                            mouth_plane.append(depth_point)
                            cv2.circle(image,(x,y),1,[0,255,0],-1 )


                        if id in all_list:
                            face.append([x, y])
                            diatance = self.get_depth(x, y, depth_frame)


                            ##print(x,y)
                            ## diatance=round(diatance,2)
                            ## print(diatance) 
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], diatance)
                            #depth_point = np.round(depth_point, 3)
                            face_plane.append(depth_point)
                            # 在人中处周围额外添加八个点********************************(李博)
                            # if id == 164: 
                            #     for myi in [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]:
                            #         diatance = self.get_depth(myi[0]+x, y+myi[1], depth_frame)
                            #         depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x+myi[0], y+myi[1]], diatance)
                            # #depth_point = np.round(depth_point, 3)
                            #         face_plane.append(depth_point)
                            #********************************************
                            #     diatance = self.get_depth(x+1, y+1, depth_frame)
                            #     depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x+1, y+1], diatance)
                            # #depth_point = np.round(depth_point, 3)
                            #     face_plane.append(depth_point)

                    # ============================= 嘴部点云 =================================
                    # pointcloud = o3d.geometry.PointCloud()
                    ## print(mouth_plane)
                        

                    np_pcd1 = np.asarray(face_plane).reshape((-1, 3)) 
                    # print("once")
                    # print(len(np_pcd1))
                    # print(np_pcd1)
                    
                    mydata=my_require1(np_pcd1)
                    print(mydata[0][0])
                    print(np_pcd1[0][0])

                    
                    # np.savetxt("face.txt",np_pcd1)

                                              # shape--> (478,3)
                    pointcloud.points = o3d.utility.Vector3dVector(np_pcd1)                     # 如果使用numpy数组可省略上两行
                    # 修改kdtree搜索邻域的距离*******************(李博)
                    ##pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
                    #*****************************************
                    pointcloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


                    # --------------------------- 点云实时可视化 -------------------------------
                    vis.update_geometry(pointcloud)
                    if to_reset:
                        vis.reset_view_point(True)
                        to_reset = False
                    vis.poll_events()
                    vis.update_renderer()



                    # normal_mouth = pointcloud.normals[164]+pointcloud.normals[0]        # 人中 处的点的三维法向
                    ##normal_mouth = pointcloud.normals[164]

                    normal_mouth=np.array([mydata[164][0], mydata[164][1],mydata[164][2]])

                    # normal_mouth/=np.linalg.norm(normal_mouth)

                    # normal_mouth[0]=-normal_mouth[0]
                    # normal_mouth[1]=-normal_mouth[1]
                    # normal_mouth[2]=-normal_mouth[2]
                    
                   
                    ##normal_mouth= compute_pionts_normal(mouth_plane)

                    normal_mouth = (normal_mouth / 10 * np.linalg.norm(normal_mouth))
                    # print("once") 
                    # print(normal_mouth) 

                    if normal_mouth[2]<0:
                        normal_mouth = -normal_mouth
                    ##print(normal_mouth)
                    # print("estimated mouth normals:", normal_mouth)

                    pp1 = np.array(rs.rs2_project_point_to_pixel(
                        depth_intrin, [normal_mouth[0], -normal_mouth[1], normal_mouth[2]]))

                    pp1 = self.filter(pp1, 10)
                    pp1 = (int(pp1[0]), int(pp1[1]))

                    rect = cv2.minAreaRect(np.array(cnts))
                    # 求出 cnts点集(41个嘴部特征点) 下的最小面积矩形
                    # (center(x,y), (width, height), angle of rotation) = cv2.minAreaRect(points)

                    mid_x, mid_y = int(rect[0][0]), int(rect[0][1])                                   # 嘴部中心点 的 x 和 y
                    mid_z = self.get_depth(mid_x, mid_y, depth_frame)

                    mouth_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [mid_x, mid_y], mid_z)  # 嘴部中心点的 三维空间坐标列表（相机坐标系下） [x,y,z]

                    end_time = time.time()
                    fps = int(1 / (end_time - start_time))
                    images = np.hstack((image, depth_colormap))


                    self.cv_show(images, rect, pp1, fps, mouth_xyz)                                   # 图片中可视化嘴部法向量

        except KeyboardInterrupt:
            pipe.stop()



def main():
    compute_pionts_normal(mydata)
    d435 = Detection()
    d435.run()

if __name__ == "__main__":
    main()