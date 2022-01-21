#ifndef SAR_PLUGIN_H
#define SAR_PLUGIN_H

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>

#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <gazebo/common/common.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
//#include <gazebo/sensors/sensors.hh>

//#include <rosflight_msgs/GNSS.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/NavSatFix.h>
//#include <geometry_msgs/TwistStamped.h>

#include <georegistration_testbed/SARCameraView.h>

namespace rosradar_plugins {

    class SARPlugin : public gazebo::ModelPlugin {

    public:
        SARPlugin();
        ~SARPlugin();

    protected:
        void Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf);
        void OnUpdate(const gazebo::common::UpdateInfo&);

    private:
        // ROS Stuff
        std::string namespace_;
        ros::NodeHandle nh_;
        ros::NodeHandle nh_private_;
        //ros::Publisher GNSS_pub_;
        ros::Publisher GNSS_fix_pub_;
        //ros::Publisher GNSS_vel_pub_;
        ros::Publisher SAR_camera_view_pub_;
        ros::Publisher SAR_ground_truth_image_pub_;
        ros::Publisher SAR_image_pub_;

        // Gazebo connections
        std::string link_name_;
        std::string ground_plane_model_name_;
        std::string sar_texture_filename_;
        gazebo::physics::WorldPtr world_;
        gazebo::physics::ModelPtr model_;
        gazebo::physics::ModelPtr ground_plane_model_;
        gazebo::physics::LinkPtr link_;
        gazebo::physics::LinkPtr link_antenna_pose_;
        gazebo::event::ConnectionPtr updateConnection_;
        gazebo::common::Time last_time_;

        gazebo::physics::PlaneShape *plane;
        ignition::math::Pose3d plane_pose;
        cv::Vec3d plane_normal;
        cv::Vec2d plane_dimensions;
        // Random Engine
        std::default_random_engine random_generator_;
        std::normal_distribution<double> standard_normal_distribution_;

        // Topic
        //std::string gnss_topic_;
        //std::string gnss_vel_topic_;
        //std::string gnss_fix_topic_;
        std::string sar_camera_view_topic_;
        std::string sar_image_topic_;
        std::string sar_truth_image_topic_;
        
        // Message with static info prefilled
        //rosflight_msgs::GNSS gnss_message_;
        sensor_msgs::NavSatFix gnss_fix_message_;
        //geometry_msgs::TwistStamped gnss_vel_message_;

        sensor_msgs::CameraInfo sar_camerainfo_;
        georegistration_testbed::SARCameraView sar_cameraview_;

        cv::Mat sar_reference_image_cv_;
        bool hasReferenceImage;

        double sar_camera_focal_length_;
        double sar_camera_fov_x_;
        double sar_camera_fov_y_;
        double sar_camera_fx_;
        double sar_camera_fy_;
        int sar_image_resolution_x_;
        int sar_image_resolution_y_;
        
        // params
        double pub_rate_;
        
        bool noise_on_;
        double x_stddev_;
        double y_stddev_;
        double z_stddev_;
        double roll_stddev_;
        double pitch_stddev_;
        double yaw_stddev_;
        double velocity_stddev_;

        double x_error_;
        double y_error_;
        double z_error_;
        double roll_error_;
        double pitch_error_;
        double yaw_error_;

        //double initial_latitude_;
        //double initial_longitude_;
        //double initial_altitude_;

        //double length_latitude_;
        //double length_longitude_;

        double sample_time_;


        void gzModelPoseToOpenCV(gazebo::physics::EntityPtr _model, cv::Mat& orientation, cv::Mat& position);

        bool intersectRayPolygon(cv::Mat rp0, cv::Mat rp1, cv::Mat tp1, cv::Mat tp2, cv::Mat tp3,
                cv::Mat& ip, bool backface_culling = false);

        void convertNorthEastToWGS84(double dpn, double dpe, double & dlat, double & dlon);

        inline double deg_to_rad(double deg) {
            return deg * M_PI / 180.0;
        }

    };
}

#endif // SAR_PLUGIN_H
