#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// int main(int argc, char** argv)
// 是UNIX、Linux以及Mac OS操作系统中C/C++的main函数标准写法
// 并且是血统最纯正的main函数写法
// 第一个参数，int型的argc，为整型，
//      用来统计程序运行时发送给main函数的命令行参数的个数，在VS中默认值为1。 
// 第二个参数，char*型的argv[]，为字符串数组，
//      用来存放指向的字符串参数的指针数组，每一个元素指向一个参数。各成员含义如下：
// argv[0]指向程序运行的全路径名 
// argv[1]指向在DOS命令行中执行程序名后的第一个字符串 
// argv[2]指向执行程序名后的第二个字符串 
// argv[3]指向执行程序名后的第三个字符串 
// argv[argc]为NULL 
int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }

    // 读取图片

// Mat本质上是由两个数据部分组成的类： 
//（包含信息有矩阵的大小，用于存储的方法，矩阵存储的地址等） 的矩阵头和一个指针，
// 指向包含了像素值的矩阵（可根据选择用于存储的方法采用任何维度存储数据）

    Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_COLOR);

    // cout<<"M="<<endl<<img_1<<endl;

    // 初始化

// KeyPoint这数据结构中有如下数据结构：
// angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
// class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
// octave：代表是从金字塔哪一层提取的得到的数据。
// pt：关键点点的坐标
// response：响应程度，代表该点强壮大小，——response代表着该关键点how good，更确切的说，是该点角点的程度。
// size：该点直径的大小

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

/*
参数说明：
    nfeatures:    保留提取的最大特征数
    scaleFactor:  金字塔抽取比率，大于1。经典金字塔默认值为2，每个层级  之间为4个像素插值关系。尺度变化较大会影响匹配性能，同时尺度范围太接近要覆盖一定的尺度，需要更多的金字塔层数，但是速度会降低。
    nlevels: 金字塔层次数量。
    edgeThreshold: 没有检测到特征的边界大小。
    firstLevel: 在当下实现中值为0.
    WTA_K: 方向BRIEF描述子产生每个元素点的数量。
    scoreType: 默认采取Harris_Score来对特征进行排序，采取Fast_Score可能 会产生特征点不是太稳定，但是计算速度快。
    matchSize:方向BRIEF描述符的区域大小
    fastThreahold: FAST角点阈值
*/
    Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // 第一步：检测Oriented FAST角点位置
    orb->detect(img_1, keypoints_1);
    orb->detect(img_2, keypoints_2);

    // 第二步：根据角点位置计算BRIEF描述子
    orb->compute( img_1, keypoints_1, descriptors_1);
    orb->compute( img_2, keypoints_2, descriptors_2);

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB特征点", outimg1);

    // 第三步：对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离

// DMatch主要用来储存匹配信息的结构体，query是要匹配的描述子，train是被匹配的描述子


    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors_1, descriptors_2, matches);

    // 第四步：匹配点对筛选
    double min_dist=10000, max_dist=0;

    // 找出所有匹配之间的最小距离和最大距离， 即最相似的和最不相似的两组点之间的距离
    for(int i=0; i<descriptors_1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist<min_dist) min_dist = dist;
        if(dist>max_dist) max_dist=dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时， 即认为匹配有误
    // 但有时候最小距离会非常小， 设置一个经验值作为下限
    vector<DMatch> good_matches;
    for(int i=0; i<descriptors_1.rows; i++)
    {
        if(matches[i].distance <= max(2*min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步：绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("所有匹配点对", img_match);
    imshow("优化后的匹配点对", img_goodmatch);
    waitKey(0);

    return 0;
}
