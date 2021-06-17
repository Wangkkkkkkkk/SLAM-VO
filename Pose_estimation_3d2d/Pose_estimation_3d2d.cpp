#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
    const Mat& img_1,
    const Mat& img_2,
    vector<KeyPoint>& keypoints_1,
    vector<KeyPoint>& keypoints_2,
    vector<DMatch>& matches
);

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    Mat& K,
    Mat& R,
    Mat& t
);

int main(int argc, char** argv){
    if(argc != 5){
        cout<<"usage:pose_estimation_3d2d img1 img2 img1_depth img2_depth"<<endl;
        return 0;
    }

    // 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> Matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, Matches);
    cout<<"一共找到 "<< Matches.size()<<" 个匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);    // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double> (3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // 相机内参
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for( DMatch m : Matches ){
        // mat.ptr<type>(row)[col]
        // Mat的ptr函数，返回的是<>中的模板类型指针，指向的是()中的第row行的起点，然后再用该指针去访问对应col列位置的元素
        
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if(d == 0){
            continue;  // 去除坏的深度点
        }
        float dd = d/1000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);

// Point3f, Point2d
// 数字（1，2，3）代表的是这个点的维度信息，
// 字母（i, f, d, l）代表该点的类型，整形、浮点型、双精度、long （例如，long int 长整形）

        pts_3d.push_back(Point3f(p1.x*dd, p1.y*dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout<<"3d-2d pairs: "<<pts_3d.size()<<endl;

    // 调用PNP求解，可选择EPNP、DLS等方法
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    Mat R;
    Rodrigues(r, R);  // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<r<<endl;
    cout<<"t="<<endl<<t<<endl;

    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
}

void find_feature_matches(const Mat& img_1, const Mat& img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch>& matches){
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);

    Mat descriptors_1, descriptors_2;
    orb->compute(img_1, keypoints_1, descriptors_1);
    orb->compute(img_2, keypoints_2, descriptors_2);

    vector<DMatch> bad_matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, bad_matches);

    double min_dist=10000, max_dist=0;
    for(int i=0; i<descriptors_1.rows; i++)
    {
        double dist = bad_matches[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist=dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    for(int i=0; i<descriptors_1.rows; i++)
    {
        if(bad_matches[i].distance <= max(2*min_dist, 30.0))
        {
            matches.push_back(bad_matches[i]);
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K ){
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment( const vector< Point3f > points_3d, const vector< Point2f > points_2d, Mat& K, Mat& R, Mat& t ){
    // 初始化g2o
    // 每个误差项优化变量维度为6，误差值维度为3
    typedef g2o::BlockSolver< g2o::BlockSolverTraits< 6, 3 > > Block;
    // 第一步：创建一个线性求解器 LinearSolver
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse< Block::PoseMatrixType >();
    // 第二步：矩阵块求解器
    Block* solver_ptr = new Block( linearSolver );
    // 梯度下降方法，从GN, LM（本次使用）, DogLeg 中选
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( solver_ptr );
    g2o::SparseOptimizer optimizer;  // 稀疏 优化模型
    optimizer.setAlgorithm( solver ); // 设置求解器

    // vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate( g2o::SE3Quat(
        R_mat,
        Eigen::Vector3d( t.at<double>(0, 0), t.at<double>(1,0), t.at<double>(2, 0))
    ));
    optimizer.addVertex( pose );

    int index = 1;
    for( const Point3f p : points_3d) {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId( index++ );
        point->setEstimate( Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized( true );
        optimizer.addVertex( point );
    }

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0, 0), Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)), 0
    );
    camera->setId(0);
    optimizer.addParameter( camera );

    // edges
    index = 1;
    for(const Point2f p:points_2d){
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId( index );
        edge->setVertex(0, dynamic_cast< g2o::VertexSBAPointXYZ* >(optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement( Eigen::Vector2d(p.x, p.y));
        edge->setParameterId(0, 0);
        edge->setInformation( Eigen::Matrix2d::Identity() );
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose( true );  // 打开调试输出
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast< chrono::duration<double> >(t2-t1);
    cout<<"optimization costs time: "<<time_used.count()<<" seconds. "<<endl;

    cout<<endl<<"after optimization: "<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;
}
