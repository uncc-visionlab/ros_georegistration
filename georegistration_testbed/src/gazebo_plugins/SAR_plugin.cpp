/*
 * Copyright 2015 James Jackson MAGICC Lab, BYU, Provo, UT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cv_bridge/cv_bridge.h>

#include "boost/endian/conversion.hpp"

#include "georegistration_testbed/SAR_plugin.h"
#include "georegistration_testbed/gz_compat.h"
#include <gazebo/common/Image.hh>
#include <gazebo/rendering/rendering.hh>
#include <ignition/math/Pose3.hh>


#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Vector3.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>

#define DEBUG 0

namespace rosradar_plugins {

    SARPlugin::SARPlugin() : ModelPlugin(), hasReferenceImage(false) {
        sar_cameraview_.homography.resize(9);
        sar_cameraview_.homography_w_noise.resize(9);
        sar_cameraview_.plane_uv_coords.resize(4 * 2);
        sar_cameraview_.plane_xyz_coords.resize(4 * 3);
        sar_cameraview_.K.resize(3 * 3);
        sar_cameraview_.pose.resize(4 * 4);
        sar_cameraview_.pose_w_noise.resize(4 * 4);
    }

    SARPlugin::~SARPlugin() {
        GZ_COMPAT_DISCONNECT_WORLD_UPDATE_BEGIN(updateConnection_);
        ROS_DEBUG_STREAM_NAMED("SAR", "Unloaded");
        nh_.shutdown();
    }

    void SARPlugin::Load(gazebo::physics::ModelPtr _model, sdf::ElementPtr _sdf) {
        // Make sure the ROS node for Gazebo has already been initialized
        if (!ros::isInitialized()) {
            ROS_FATAL("A ROS node for Gazebo has not been initialized, unable to load SAR plugin");
            return;
        }
        ROS_INFO("Loaded the SAR plugin");

        //
        // Configure Gazebo Integration
        //

        //
        // Get elements from the robot urdf/sdf file
        //
        model_ = _model;
        world_ = model_->GetWorld();

        last_time_ = GZ_COMPAT_GET_SIM_TIME(world_);

        namespace_.clear();

        if (_sdf->HasElement("namespace"))
            namespace_ = _sdf->GetElement("namespace")->Get<std::string>();
        else
            ROS_ERROR_NAMED("SAR_plugin", "Please specify a namespace.");

        if (_sdf->HasElement("linkName"))
            link_name_ = _sdf->GetElement("linkName")->Get<std::string>();
        else
            ROS_ERROR_NAMED("SAR_plugin", "Please specify a linkName.");

        gazebo::physics::Link_V links = model_->GetLinks();
        for (auto &link : links) {
            ROS_ERROR_NAMED("SAR_plugin", "Link = %s", link->GetName().c_str());
            //gazebo::physics::LinkPtr sar_link = link->GetChildLink(namespace_ + "_sar_optical_frame");
            //ROS_ERROR_NAMED("SAR_plugin", "Link = %s", sar_link->GetName().c_str());
        }
        gazebo::physics::Joint_V joints = model_->GetJoints();
        for (auto &joint : joints) {
            ROS_ERROR_NAMED("SAR_plugin", "Joint = %s", joint->GetName().c_str());
        }

        link_ = model_->GetLink(link_name_);
        if (link_ == nullptr)
            gzthrow("[SAR_plugin] Couldn't find specified link \"" << link_name_ << "\".");

        link_antenna_pose_ = model_->GetLink(namespace_ + "/sar_optical_frame");
        if (link_antenna_pose_ == nullptr)
            gzthrow("[SAR_plugin] Couldn't find specified link \"" << namespace_ + "/sar_optical_frame" << "\".");
        //
        // ROS Node Setup
        //
        nh_ = ros::NodeHandle(namespace_);
        nh_private_ = ros::NodeHandle(namespace_ + "/sar");

        // load params from rosparam server
        noise_on_ = nh_private_.param<bool>("noise_on", true);
        pub_rate_ = nh_private_.param<double>("rate", 10.0);
        //gnss_topic_ = nh_private_.param<std::string>("topic", "sar");
        //gnss_fix_topic_ = nh_private_.param<std::string>("fix_topic", "sar/fix");
        sar_camera_focal_length_ = nh_private_.param<double>("sar_camera_focal_length", 12.0e-3);
        sar_camera_fov_x_ = nh_private_.param<int>("sar_camera_field_of_view_x", 30);
        sar_camera_fov_y_ = nh_private_.param<int>("sar_camera_field_of_view_y", 30);
        sar_camera_fx_ = nh_private_.param<double>("sar_camera_fx", 530.0);
        sar_camera_fy_ = nh_private_.param<double>("sar_camera_fy", 530.0);
        sar_image_resolution_x_ = nh_private_.param<int>("sar_camera_resolution_x", 512);
        sar_image_resolution_y_ = nh_private_.param<int>("sar_camera_resolution_y", 512);
        sar_camera_view_topic_ = nh_private_.param<std::string>("sar_camera_view_topic", "sar/camera_view");
        sar_image_topic_ = nh_private_.param<std::string>("sar_image_topic", "sar/image_raw");
        sar_truth_image_topic_ = nh_private_.param<std::string>("sar_truth_image_topic", "sar/truth/image_raw");
        ground_plane_model_name_ = nh_private_.param<std::string>("ground_plane_model_name", "");
        sar_texture_filename_ = nh_private_.param<std::string>("sar_texture_filename", "");
        //gnss_vel_topic_ = nh_private_.param<std::string>("vel_topic", "sar/vel");
        x_stddev_ = nh_private_.param<double>("x_stddev", 0.1);
        y_stddev_ = nh_private_.param<double>("y_stddev", 0.1);
        z_stddev_ = nh_private_.param<double>("z_stddev", 0.1);
        velocity_stddev_ = nh_private_.param<double>("velocity_stddev", 0.30);
        roll_stddev_ = nh_private_.param<double>("roll_stddev", 0.01);
        pitch_stddev_ = nh_private_.param<double>("pitch_stddev", 0.01);
        yaw_stddev_ = nh_private_.param<double>("yaw_stddev", 0.01);
        //initial_latitude_ = nh_private_.param<double>("initial_latitude", 40.267320); // default to Provo, UT
        //initial_longitude_ = nh_private_.param<double>("initial_longitude", -111.635629); // default to Provo, UT
        //initial_altitude_ = nh_private_.param<double>("initial_altitude", 1387.0); // default to Provo, UT

        // ROS Publishers
        //GNSS_pub_ = nh_.advertise<rosflight_msgs::GNSS>(gnss_topic_, 1);
        //GNSS_fix_pub_ = nh_.advertise<sensor_msgs::NavSatFix>(gnss_fix_topic_, 1);
        //GNSS_vel_pub_ = nh_.advertise<geometry_msgs::TwistStamped>(gnss_vel_topic_, 1);
        SAR_ground_truth_image_pub_ = nh_.advertise<sensor_msgs::Image>(sar_truth_image_topic_, 1);
        SAR_camera_view_pub_ = nh_.advertise<georegistration_testbed::SARCameraView>(sar_camera_view_topic_, 1);
        SAR_image_pub_ = nh_.advertise<sensor_msgs::Image>(sar_image_topic_, 1);

        if (ground_plane_model_name_ != "") {
            //ground_plane_model_name_ = _sdf->GetElement("modelName")->Get<std::string>();
#if GAZEBO_MAJOR_VERSION >= 8
            ground_plane_model_ = world_->ModelByName(ground_plane_model_name_);
#else
            ground_plane_model_ = world_->GetModel(ground_plane_model_name_);
#endif
            if (!ground_plane_model_) {
                ROS_ERROR_NAMED("SAR_plugin", "Couldn't find specified ground plane model \"%s\".", ground_plane_model_name_.c_str());
                ROS_ERROR_NAMED("SAR_plugin", "ModelByName: model [%s] does not exist", ground_plane_model_name_.c_str());
            } else {
                ROS_ERROR_NAMED("SAR_plugin", "Got ground plane model \"%s\".", ground_plane_model_name_.c_str());
                plane_pose = ground_plane_model_->WorldPose();
                // get model parent name
                gazebo::physics::ModelPtr parent_model = boost::dynamic_pointer_cast<gazebo::physics::Model>(ground_plane_model_->GetParent());

                std::string parent_model_name;
                if (parent_model)
                    parent_model_name = parent_model->GetName();

                // get list of child bodies, geoms
                std::vector<std::string> body_names, geom_names;
                std::vector<geometry_msgs::Vector3> geom_sizes, geom_scales;
                body_names.clear();
                geom_names.clear();
                for (unsigned int i = 0; i < ground_plane_model_->GetChildCount(); i++) {
                    gazebo::physics::LinkPtr body = boost::dynamic_pointer_cast<gazebo::physics::Link>(ground_plane_model_->GetChild(i));
                    if (body) {
                        body_names.push_back(body->GetName());
                        // get list of geoms
                        for (unsigned int j = 0; j < body->GetChildCount(); j++) {
                            gazebo::physics::CollisionPtr geom = boost::dynamic_pointer_cast<gazebo::physics::Collision>(body->GetChild(j));
                            if (geom) {
                                geom_names.push_back(geom->GetName());
                                //res.set_geom_name(geom->GetName());
                                gazebo::physics::ShapePtr shape(geom->GetShape());
                                std::cout << "Found geometry " << std::endl;
                                if (shape->HasType(gazebo::physics::Base::BOX_SHAPE)) {
                                    gazebo::physics::BoxShape *box = static_cast<gazebo::physics::BoxShape*> (shape.get());
                                    ignition::math::Vector3d tmp_size = box->Size();
                                    ignition::math::Vector3d tmp_scale = box->Scale();
                                    geometry_msgs::Vector3 geom_size = geometry_msgs::Vector3();
                                    geom_size.x = tmp_size[0];
                                    geom_size.y = tmp_size[1];
                                    geom_size.z = tmp_size[2];
                                    geom_sizes.push_back(geom_size);
                                    geometry_msgs::Vector3 geom_scale = geometry_msgs::Vector3();
                                    geom_size.x = tmp_scale[0];
                                    geom_size.y = tmp_scale[1];
                                    geom_size.z = tmp_scale[2];
                                    geom_scales.push_back(geom_scale);
                                    std::cout << "Set BOX message data!" << std::endl;
                                } else if (shape->HasType(gazebo::physics::Base::PLANE_SHAPE)) {
                                    plane = static_cast<gazebo::physics::PlaneShape*> (shape.get());
                                    ignition::math::Vector2d ign_plane_size = plane->Size();
                                    plane_dimensions[0] = ign_plane_size.X();
                                    plane_dimensions[1] = ign_plane_size.Y();
                                    ROS_ERROR_NAMED("SAR_plugin", "Found PLANE parameters!");
                                    ignition::math::Vector3d ign_plane_normal = plane->Normal();
                                    plane_normal[0] = ign_plane_normal.X();
                                    plane_normal[1] = ign_plane_normal.Y();
                                    plane_normal[2] = ign_plane_normal.Z();
                                }
                            }
                        }
                    }
                }
            }
        } else {
            ROS_ERROR_NAMED("SAR_plugin", "Please specify a modelName.");
        }

        if (sar_texture_filename_ != "") {
            ROS_ERROR_NAMED("SAR_plugin", "Got texture %s.", sar_texture_filename_.c_str());
            sar_reference_image_cv_ = cv::imread(sar_texture_filename_);
            ROS_ERROR_NAMED("SAR_plugin", "reference image has size %dx%d.",
                    sar_reference_image_cv_.cols, sar_reference_image_cv_.rows);
            hasReferenceImage = true;
        } else {
            ROS_ERROR_NAMED("SAR_plugin", "Please specify a textureFilename.");
        }


        // Calculate sample time from sensor update rate
        sample_time_ = 1.0 / pub_rate_;

        // Configure Noise
        random_generator_ = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        standard_normal_distribution_ = std::normal_distribution<double>(0.0, 1.0);

        // disable noise by zeroing the standard deviation of the noise
        if (!noise_on_) {
            x_stddev_ = 0;
            y_stddev_ = 0;
            z_stddev_ = 0;
            velocity_stddev_ = 0;
            roll_stddev_ = 0;
            pitch_stddev_ = 0;
            yaw_stddev_ = 0;
        }

        // Fill static members of SAR message.
        //gnss_message_.header.frame_id = link_name_;
        //TODO add constants for UBX fix types
        //gnss_message_.fix = 3; // corresponds to a 3D fix
        gnss_fix_message_.status.service = sensor_msgs::NavSatStatus::SERVICE_GPS;
        gnss_fix_message_.status.status = sensor_msgs::NavSatStatus::STATUS_FIX;

        // initialize SAR error to zero
        x_error_ = 0.0;
        y_error_ = 0.0;
        z_error_ = 0.0;
        roll_error_ = 0.0;
        pitch_error_ = 0.0;
        yaw_error_ = 0.0;

        // Listen to the update event. This event is broadcast every simulation iteration.
        updateConnection_ = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&SARPlugin::OnUpdate, this, std::placeholders::_1));
    }

    // This gets called by the world update start event.

    void SARPlugin::OnUpdate(const gazebo::common::UpdateInfo & _info) {
        // check if time to publish
        gazebo::common::Time current_time = GZ_COMPAT_GET_SIM_TIME(world_);
        if ((current_time - last_time_).Double() >= sample_time_) {
            last_time_ = current_time;

            if (!hasReferenceImage) {
                return;
            }

            cv::Mat_<double> antenna_orientation(3, 3);
            cv::Mat_<double> antenna_position(3, 1);
            //gzModelPoseToOpenCV(link_, antenna_orientation, antenna_position);
            gzModelPoseToOpenCV(link_antenna_pose_, antenna_orientation, antenna_position);


            cv::Mat antennaXVec = antenna_orientation.col(0);
            // the antennaYVec and antennaZVec determine the image plane (X,Y) axes in 3D
            cv::Mat antennaYVec = antenna_orientation.col(2);
            cv::Mat antennaZVec = antenna_orientation.col(1);

            cv::Mat_<double> camera_orientation(3, 3);
            camera_orientation.row(0) = antennaZVec.t();
            camera_orientation.row(1) = antennaYVec.t();
            camera_orientation.row(2) = antennaXVec.t();


            // compute ground truth pose
            for (int c = 0; c < camera_orientation.cols; c++) {
                for (int r = 0; r < camera_orientation.rows; r++) {
                    sar_cameraview_.pose[r * 4 + c] = camera_orientation.at<double>(r, c);
                }
            }
            for (int r = 0; r < antenna_position.rows; r++) {
                sar_cameraview_.pose[r * 4 + 3] = antenna_position.at<double>(r, 0);
            }
            sar_cameraview_.pose[15] = 1.0;

            // compute pose with user-configured XYZ position and orientation noise 
            cv::Mat_<double> noiseRPY_orientation = camera_orientation.clone();
            //cv::Mat_<double> noiseRPY_orientation = cv::Mat_<double>::zeros(3, 3);
            double noise_roll = roll_stddev_ * standard_normal_distribution_(random_generator_) * M_PI / 180.0;
            double noise_pitch = pitch_stddev_ * standard_normal_distribution_(random_generator_) * M_PI / 180.0;
            double noise_yaw = yaw_stddev_ * standard_normal_distribution_(random_generator_) * M_PI / 180.0;
            noiseRPY_orientation.at<double>(0, 1) += -noise_yaw;
            noiseRPY_orientation.at<double>(1, 0) += noise_yaw;
            noiseRPY_orientation.at<double>(0, 2) += noise_pitch;
            noiseRPY_orientation.at<double>(2, 0) += -noise_pitch;
            noiseRPY_orientation.at<double>(1, 2) += -noise_roll;
            noiseRPY_orientation.at<double>(2, 1) += noise_roll;

            cv::SVD rotationSVD(noiseRPY_orientation, cv::SVD::FULL_UV); // constructor
            for (int r = 0; r < rotationSVD.w.rows; r++) {
                //std::cout << " replaced W diagonal value " << rotationSVD.w.at<double>(r) << " with 1." << std::endl;
                rotationSVD.w.at<double>(r) = 1.0;
            }
            noiseRPY_orientation = rotationSVD.u * cv::Mat::diag(rotationSVD.w) * rotationSVD.vt;
            //std::cout << "noiseRPY" << noiseRPY_orientation << std::endl;
            for (int c = 0; c < noiseRPY_orientation.cols; c++) {
                for (int r = 0; r < noiseRPY_orientation.rows; r++) {
                    sar_cameraview_.pose_w_noise[r * 4 + c] = noiseRPY_orientation.at<double>(r, c);
                }
            }

            // Add noise per Gauss-Markov Process (p. 139 UAV Book)
            cv::Mat_<double> noiseXYZ_position(3, 1);
            noiseXYZ_position.at<double>(0, 0) = x_stddev_ * standard_normal_distribution_(random_generator_);
            //north_SAR_error_ = exp(-1.0 * north_k_SAR_ * sample_time_) * north_SAR_error_ + noise*sample_time_;
            noiseXYZ_position.at<double>(1, 0) = y_stddev_ * standard_normal_distribution_(random_generator_);
            //east_SAR_error_ = exp(-1.0 * east_k_SAR_ * sample_time_) * east_SAR_error_ + noise*sample_time_;
            noiseXYZ_position.at<double>(2, 0) = z_stddev_ * standard_normal_distribution_(random_generator_);
            //alt_SAR_error_ = exp(-1.0 * alt_k_SAR_ * sample_time_) * alt_SAR_error_ + noise*sample_time_;
            for (int r = 0; r < antenna_position.rows; r++) {
                sar_cameraview_.pose_w_noise[r * 4 + 3] = antenna_position.at<double>(r, 0) + noiseXYZ_position.at<double>(r, 0);
            }
            sar_cameraview_.pose_w_noise[15] = 1.0;

            // get the pose of the ground plane model
            cv::Mat_<double> ground_plane_orientation(3, 3);
            cv::Mat_<double> ground_plane_position(3, 1);
            gzModelPoseToOpenCV(ground_plane_model_, ground_plane_orientation, ground_plane_position);
            // in Gazebo X = North, Y = West, Z = Up
            cv::Mat orientedNormal = ground_plane_orientation * cv::Mat(plane_normal);
            cv::Vec3d plane_u(1.0, 0.0, 0.0);
            plane_u = plane_u - plane_u.dot(plane_normal) * plane_normal;
            plane_u = plane_u / cv::norm(plane_u);
            cv::Mat orientedUVec = ground_plane_orientation * cv::Mat(plane_u);
            cv::Vec3d plane_v(0.0, -1.0, 0.0);
            plane_v = plane_v - plane_v.dot(plane_normal) * plane_normal;
            plane_v = plane_v / cv::norm(plane_v);
            cv::Mat orientedVVec = ground_plane_orientation * cv::Mat(plane_v);

            // calculate 3D positions of the corners of the plane model
            cv::Mat_<double> plane_corners(3, 4);
            plane_corners.col(0) = ground_plane_position - 0.5 * plane_dimensions[0] * orientedUVec - 0.5 * plane_dimensions[1] * orientedVVec;
            plane_corners.col(1) = ground_plane_position + 0.5 * plane_dimensions[0] * orientedUVec - 0.5 * plane_dimensions[1] * orientedVVec;
            plane_corners.col(2) = ground_plane_position - 0.5 * plane_dimensions[0] * orientedUVec + 0.5 * plane_dimensions[1] * orientedVVec;
            plane_corners.col(3) = ground_plane_position + 0.5 * plane_dimensions[0] * orientedUVec + 0.5 * plane_dimensions[1] * orientedVVec;

            // calculate coefficients of the algebraic plane containing the plane model
            cv::Mat_<double> plane_coefficients(4, 1);
            plane_coefficients.at<double>(0, 0) = plane_normal[0];
            plane_coefficients.at<double>(1, 0) = plane_normal[1];
            plane_coefficients.at<double>(2, 0) = plane_normal[2];
            plane_coefficients.at<double>(3, 0) = -plane_normal.dot(ground_plane_position);

            cv::Mat pixel_size = (cv::Mat_<double>(2, 1) << sar_camera_focal_length_ / sar_camera_fx_, sar_camera_focal_length_ / sar_camera_fy_);
            cv::Mat xy_resolution = (cv::Mat_<double>(2, 1) << sar_image_resolution_x_, sar_image_resolution_y_);
            sar_cameraview_.K[0] = sar_camera_fx_;
            sar_cameraview_.K[4] = sar_camera_fy_;
            sar_cameraview_.K[2] = sar_image_resolution_x_ / 2;
            sar_cameraview_.K[5] = sar_image_resolution_y_ / 2;
            sar_cameraview_.K[8] = 1.0;
            cv::Mat sensor_physical_dims = (cv::Mat_<double>(2, 1)
                    << pixel_size.at<double>(0) * xy_resolution.at<double>(0),
                    pixel_size.at<double>(1) * xy_resolution.at<double>(1));

            // convert pixel coords to physical coords
            //cv::Mat sar_sensor_phase_center_position = (cv::Mat_<double>(3, 1) << pn, pe, h);
            cv::Mat_<double> sar_sensor_corners_physical(3, 4);
            //sar_sensor_corners_physical.col(0) = antenna_position - 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec - 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            //sar_sensor_corners_physical.col(1) = antenna_position + 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec - 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            //sar_sensor_corners_physical.col(2) = antenna_position - 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec + 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            //sar_sensor_corners_physical.col(3) = antenna_position + 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec + 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            sar_sensor_corners_physical.col(0) = antenna_position + 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec + 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            sar_sensor_corners_physical.col(1) = antenna_position - 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec + 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            sar_sensor_corners_physical.col(2) = antenna_position + 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec - 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;
            sar_sensor_corners_physical.col(3) = antenna_position - 0.5 * sensor_physical_dims.at<double>(0) * antennaZVec - 0.5 * sensor_physical_dims.at<double>(1) * antennaYVec + sar_camera_focal_length_*antennaXVec;

            //cv::Mat_<double> canonicalToGazebo(3,3);
            //double angle = M_PI/2.0;
            //cv::Mat rodriguesVec = (cv::Mat_<double>(3,1) << angle*antennaXVec.at<double>(0), angle*antennaXVec.at<double>(1), angle*antennaXVec.at<double>(2));
            //cv::Rodrigues(rodriguesVec, canonicalToGazebo);
            //sar_sensor_corners_physical = canonicalToGazebo*sar_sensor_corners_physical;

#if DEBUG > 2
            ROS_ERROR_NAMED("SAR_plugin", "position = (%f,%f,%f)",
                    ground_plane_position.at<double>(0, 0), ground_plane_position.at<double>(1, 0), ground_plane_position.at<double>(2, 0));
            ROS_ERROR_NAMED("SAR_plugin", "dimensions = (%f,%f)", plane_dimensions[0], plane_dimensions[1]);
            ROS_ERROR_NAMED("SAR_plugin", "Oriented normal = (%f,%f,%f) normal = (%f,%f,%f)",
                    orientedNormal.at<double>(0, 0), orientedNormal.at<double>(1, 0), orientedNormal.at<double>(2, 0),
                    plane_normal[0], plane_normal[1], plane_normal[2]);
            ROS_ERROR_NAMED("SAR_plugin", "orientedUVec = (%f,%f,%f)",
                    orientedUVec.at<double>(0, 0), orientedUVec.at<double>(1, 0), orientedUVec.at<double>(2, 0));
            ROS_ERROR_NAMED("SAR_plugin", "orientedVVec = (%f,%f,%f)",
                    orientedVVec.at<double>(0, 0), orientedVVec.at<double>(1, 0), orientedVVec.at<double>(2, 0));
            ROS_ERROR_NAMED("SAR_plugin", "c_nw = (%f,%f,%f)",
                    plane_corners.at<double>(0, 0), plane_corners.at<double>(1, 0), plane_corners.at<double>(2, 0));
            ROS_ERROR_NAMED("SAR_plugin", "c_ne = (%f,%f,%f)",
                    plane_corners.at<double>(0, 1), plane_corners.at<double>(1, 1), plane_corners.at<double>(2, 1));
            ROS_ERROR_NAMED("SAR_plugin", "c_sw = (%f,%f,%f)",
                    plane_corners.at<double>(0, 2), plane_corners.at<double>(1, 2), plane_corners.at<double>(2, 2));
            ROS_ERROR_NAMED("SAR_plugin", "c_se = (%f,%f,%f)",
                    plane_corners.at<double>(0, 3), plane_corners.at<double>(1, 3), plane_corners.at<double>(2, 3));

            ROS_ERROR_NAMED("SAR_plugin", "antenna_position = (%f,%f,%f)",
                    antenna_position.at<double>(0), antenna_position.at<double>(1), antenna_position.at<double>(2));
            //ROS_ERROR_NAMED("SAR_plugin", "noisy_antenna_position = (%f,%f,%f)",
            //        sar_sensor_phase_center_position.at<double>(0), sar_sensor_phase_center_position.at<double>(1), sar_sensor_phase_center_position.at<double>(2));
            ROS_ERROR_NAMED("SAR_plugin", "rf_propagation_axis = (%f,%f,%f)",
                    antennaXVec.at<double>(0), antennaXVec.at<double>(1), antennaXVec.at<double>(2));
            ROS_ERROR_NAMED("SAR_plugin", "antennaYVec = (%f,%f,%f)",
                    antennaYVec.at<double>(0), antennaYVec.at<double>(1), antennaYVec.at<double>(2));
            ROS_ERROR_NAMED("SAR_plugin", "antennaZVec = (%f,%f,%f)",
                    antennaZVec.at<double>(0), antennaZVec.at<double>(1), antennaZVec.at<double>(2));
#endif
            // make vector into 3d world
            cv::Mat_<double> ip(3, 1);
            cv::Mat rp0 = antenna_position;
            cv::Mat tp0 = plane_corners.col(0);
            cv::Mat tp1 = plane_corners.col(1);
            cv::Mat tp2 = plane_corners.col(2);
            cv::Mat tp3 = plane_corners.col(3);

            cv::Mat_<double> uv_texture_coords(2, 4);
            //cv::Mat_<double> uv_texture_coords_w_noise(2, 4);
            bool validHomographyEstimate = true;
            // find 3d intersection pt with plane 
            // convert 3d intersection pt to pixel coordinate in textured image
            for (int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {
                cv::Mat rp1 = sar_sensor_corners_physical.col(cornerIndex);
                if (intersectRayPolygon(rp0, rp1, tp1, tp0, tp2, ip)) {
                    uv_texture_coords.at<double>(0, cornerIndex) = orientedUVec.dot(ip - tp0) / plane_dimensions[0];
                    uv_texture_coords.at<double>(1, cornerIndex) = orientedVVec.dot(ip - tp0) / plane_dimensions[1];
                    sar_cameraview_.plane_xyz_coords[cornerIndex] = (float) ip.at<double>(0);
                    sar_cameraview_.plane_xyz_coords[cornerIndex + 4] = (float) ip.at<double>(1);
                    sar_cameraview_.plane_xyz_coords[cornerIndex + 8] = (float) ip.at<double>(2);
                    //uv_texture_coords.at<double>(0, cornerIndex) = (ip.at<double>(0) - tp0.at<double>(0)) / plane_dimensions[0];
                    //uv_texture_coords.at<double>(1, cornerIndex) = (ip.at<double>(1) - tp0.at<double>(1)) / plane_dimensions[1];
#if DEBUG > 3
                    ROS_ERROR_NAMED("SAR_plugin", "found intersection triangle[0] = (%f,%f,%f)",
                            ip.at<double>(0), ip.at<double>(1), ip.at<double>(2));
                    ROS_ERROR_NAMED("SAR_plugin", "uv[%d] = (%f,%f)",
                            cornerIndex, uv_texture_coords.at<double>(0, cornerIndex), uv_texture_coords.at<double>(1, cornerIndex));
#endif
                } else if (intersectRayPolygon(rp0, rp1, tp1, tp2, tp3, ip)) {
                    uv_texture_coords.at<double>(0, cornerIndex) = orientedUVec.dot(ip - tp0) / plane_dimensions[0];
                    uv_texture_coords.at<double>(1, cornerIndex) = orientedVVec.dot(ip - tp0) / plane_dimensions[1];
                    sar_cameraview_.plane_xyz_coords[cornerIndex] = (float) ip.at<double>(0);
                    sar_cameraview_.plane_xyz_coords[cornerIndex + 4] = (float) ip.at<double>(1);
                    sar_cameraview_.plane_xyz_coords[cornerIndex + 8] = (float) ip.at<double>(2);
                    //uv_texture_coords.at<double>(0, cornerIndex) = (ip.at<double>(0) - tp0.at<double>(0)) / plane_dimensions[0];
                    //uv_texture_coords.at<double>(1, cornerIndex) = (ip.at<double>(1) - tp0.at<double>(1)) / plane_dimensions[1];
#if DEBUG > 3
                    ROS_ERROR_NAMED("SAR_plugin", "found intersection triangle[1] = (%f,%f,%f)",
                            ip.at<double>(0), ip.at<double>(1), ip.at<double>(2));
                    ROS_ERROR_NAMED("SAR_plugin", "uv[%d] = (%f,%f)",
                            cornerIndex, uv_texture_coords.at<double>(0, cornerIndex), uv_texture_coords.at<double>(1, cornerIndex));
#endif
                } else {
                    uv_texture_coords.col(cornerIndex) = std::numeric_limits<double>::infinity();
                    validHomographyEstimate = false;
                }
            }
            if (!validHomographyEstimate) {
                return;
            }

            // find the 4 corner pixel locations in the source image
            int iWidth = sar_image_resolution_x_, iHeight = sar_image_resolution_y_;
            std::vector<cv::Point2f> sar_pixel_coords_ground_plane(4);
            std::vector<cv::Point2f> sar_pixel_coords_uv(4);
            for (int cornerIndex = 0; cornerIndex < 4; cornerIndex++) {
                sar_pixel_coords_uv[cornerIndex] = cv::Point2f((float) uv_texture_coords.at<double>(0, cornerIndex),
                        (float) uv_texture_coords.at<double>(1, cornerIndex));
                sar_pixel_coords_ground_plane[cornerIndex] = cv::Point2f((float) uv_texture_coords.at<double>(0, cornerIndex) * sar_reference_image_cv_.cols,
                        (float) uv_texture_coords.at<double>(1, cornerIndex) * sar_reference_image_cv_.rows);
                sar_cameraview_.plane_uv_coords[cornerIndex] = uv_texture_coords.at<double>(0, cornerIndex);
                sar_cameraview_.plane_uv_coords[4 + cornerIndex] = uv_texture_coords.at<double>(1, cornerIndex);
#if DEBUG > 3
                ROS_ERROR_NAMED("SAR_plugin", "sar_pixel_coords[%d] = (%f,%f)",
                        cornerIndex, sar_pixel_coords_ground_plane[cornerIndex].x, sar_pixel_coords_ground_plane[cornerIndex].y);
#endif
            }

            std::vector<cv::Point2f> sar_pixel_coords_rect_image(4);
            sar_pixel_coords_rect_image[0] = cv::Point2f(0, 0);
            sar_pixel_coords_rect_image[1] = cv::Point2f((float) iWidth - 1, 0);
            sar_pixel_coords_rect_image[2] = cv::Point2f(0, (float) iHeight - 1);
            sar_pixel_coords_rect_image[3] = cv::Point2f((float) iWidth - 1, (float) iHeight - 1);

            // call findHomography to estimate the homography
            //cv::Mat H = cv::findHomography(sar_pixel_coords_rect_image, sar_pixel_coords_ground_plane);
            cv::Mat H = cv::getPerspectiveTransform(sar_pixel_coords_ground_plane, sar_pixel_coords_rect_image);
            //cv::Mat H = cv::getPerspectiveTransform(sar_pixel_coords_rect_image, sar_pixel_coords_uv);
            if (H.cols != 3 || H.rows != 3) {
                validHomographyEstimate = false;
                return;
            }

            for (int r = 0; r < H.rows; r++) {
                for (int c = 0; c < H.cols; c++) {
                    sar_cameraview_.homography[r * H.cols + c] = H.at<double>(r, c);
                }
            }

            for (int r = 0; r < H.rows; r++) {
                for (int c = 0; c < H.cols; c++) {
                    sar_cameraview_.homography_w_noise[r * H.cols + c] = H.at<double>(r, c);
                }
            }

            sar_cameraview_.header.stamp.fromSec(GZ_COMPAT_GET_SIM_TIME(world_).Double());
            //std::cout << "H:\n" << H << std::endl;

            // call warpPerspective to map the texture to a rectangular image
            // publish the warped image
            cv::Mat simulated_sar_image_warp, simulated_sar_image_final;
            cv::warpPerspective(sar_reference_image_cv_, simulated_sar_image_warp, H, cv::Size(iHeight, iWidth));
            //cv::Mat_<double> cc90Rotation = (cv::Mat_<double>(2,3) << 0, -1, 0, 1, 0, 0);       
            //cv::Point2f imageCenter = 0.5 * sar_pixel_coords_rect_image[3];
            // rotate image counter-clockwise 90 degrees around the image center (scale 1.0)
            //cv::Mat cc90Rotation = cv::getRotationMatrix2D(imageCenter, 90.0, 1.0);
            // flip the X-axis pixels
            //cc90Rotation.at<double>(0, 0) = -cc90Rotation.at<double>(0, 0);
            //cc90Rotation.at<double>(0, 1) = -cc90Rotation.at<double>(0, 1);
            //cc90Rotation.at<double>(0, 2) = cc90Rotation.at<double>(0, 2) + iWidth;
            // compute the new image
            //cv::warpAffine(simulated_sar_image_warp, simulated_sar_image_final, cc90Rotation, simulated_sar_image_warp.size());
            simulated_sar_image_final = simulated_sar_image_warp.clone();

            cv::Mat sar_reference_image_copy = sar_reference_image_cv_.clone();
            int lineType = cv::LINE_8;

            cv::Point polyPts[4] = {cv::Point(sar_pixel_coords_ground_plane[0]), cv::Point(sar_pixel_coords_ground_plane[1]),
                cv::Point(sar_pixel_coords_ground_plane[3]), cv::Point(sar_pixel_coords_ground_plane[2])};
            cv::fillConvexPoly(sar_reference_image_copy, polyPts, 4, cv::Scalar(0, 255, 0), lineType, 0);
            //cv::namedWindow("SAR Image", cv::WINDOW_NORMAL);
            //cv::imshow("SAR Image", simulated_sar_image_final);
            //cv::resizeWindow("SAR Image", 200, 200);
            //cv::waitKey(1);

            //imshow(sar_reference_image_copy);
            cv_bridge::CvImage sar_truth_img_msg;
            sar_truth_img_msg.image = sar_reference_image_copy;
            sar_truth_img_msg.header.stamp.fromSec(GZ_COMPAT_GET_SIM_TIME(world_).Double());
            sar_truth_img_msg.encoding = sensor_msgs::image_encodings::BGR8;

            SAR_ground_truth_image_pub_.publish(sar_truth_img_msg.toImageMsg());

            cv_bridge::CvImage sar_img_msg;
            sar_img_msg.image = simulated_sar_image_final;
            sar_img_msg.header.stamp.fromSec(GZ_COMPAT_GET_SIM_TIME(world_).Double());
            sar_img_msg.encoding = sensor_msgs::image_encodings::BGR8;

            SAR_image_pub_.publish(sar_img_msg.toImageMsg());
            SAR_camera_view_pub_.publish(sar_cameraview_);
        }

    }

    void SARPlugin::gzModelPoseToOpenCV(gazebo::physics::EntityPtr _entity, cv::Mat& orientation, cv::Mat & position) {
        ignition::math::Pose3d gz_pose = _entity->WorldPose();
        ignition::math::Vector3d gz_position = gz_pose.Pos();
        ignition::math::Quaterniond quat = gz_pose.Rot();
        double normf = sqrt(1.0 - quat.W() * quat.W());
        double EPS_TOL = 1.0e-4;
        if (normf < EPS_TOL) {
            normf = 1.0;
        }
        double angle = 2 * acos(quat.W());
        cv::Vec3d axis_of_rotation(quat.X() / normf, quat.Y() / normf, quat.Z() / normf);
        //double normCheck = cv::norm(axis_of_rotation);                
        cv::Vec3d rodriguesVec = axis_of_rotation*angle;
        position = (cv::Mat_<double>(3, 1) << gz_position.X(), gz_position.Y(), gz_position.Z());
        cv::Rodrigues(rodriguesVec, orientation);
        //ROS_ERROR_NAMED("SAR_plugin", "Quaternion norm = %f", normCheck);     
    }

    //  
    //   This function intersects a line passing through the 3D points (rp0,rp1) with
    //   the triangle formed by the three sequential 3D points (tp1,tp2,tp3). Note that
    //   the implied triangular surface is defined by the 3 line segments ending with
    //   endpoints (tp1,tp2), (tp2,tp3), and (tp3,tp1). The order of the points defines
    //   orientation of the polynomial in terms of the "inward" pointing surface normal.
    //  
    //   (rp0, rp1)       - the 3D point pair through which the ray passes.
    //  
    //   (tp1,tp2,tp3)    - the 3D point ordered triplet specifying the vertices of the triangle
    //                      of the triangle to be intersected by the ray.
    //  
    //   backface_culling - this variable effects what is considered to be a valid polygon-ray
    //                      intersection. Specifically, when backface culling is set to 1 only 
    //                      triangles with normals pointing opposite the ray rp1-rp0 are 
    //                      considered for intersection. When backface culling is set to 0 
    //                      all triangles are considered for intersection.
    //   
    //   

    bool SARPlugin::intersectRayPolygon(cv::Mat rp0, cv::Mat rp1, cv::Mat tp1, cv::Mat tp2, cv::Mat tp3,
            cv::Mat& ip, bool backface_culling) {

        bool intersect = false;
        ip.at<double>(0, 0) = ip.at<double>(1, 0) = ip.at<double>(2, 0) = std::numeric_limits<double>::infinity();
        cv::Mat e1m = (tp2 - tp1);
        cv::Mat e2m = (tp3 - tp1);
        cv::Mat raym = (rp1 - rp0);
        cv::Vec3d e1((double *) e1m.data);
        cv::Vec3d e2((double *) e2m.data);
        cv::Vec3d ray((double *) raym.data);
        cv::Vec3d n = e1.cross(e2);
        n = n / cv::norm(n);

        double denominator = ray.dot(n);
        //% backface culling
        if (backface_culling == false) {
            if (abs(denominator) < 1E-10) {
                return intersect;
            }
        } else if (denominator < 1E-10) {
            return intersect;
        }
        //% no backface culling
        cv::Mat tp1rp0m = (tp1 - rp0);
        cv::Vec3d tp1rp0((double *) tp1rp0m.data);
        double tparam = tp1rp0.dot(n) / denominator;
        cv::Mat planePoint = rp0 + tparam * (rp1 - rp0);
        cv::Mat wVec = planePoint - tp1;
        cv::Mat vVec = e1m;
        cv::Mat uVec = e2m;
        cv::Mat mat1x1;
        mat1x1 = (uVec.t() * uVec);
        double ulengthsq = mat1x1.at<double>(0, 0);
        mat1x1 = (vVec.t() * vVec);
        double vlengthsq = mat1x1.at<double>(0, 0);
        mat1x1 = (uVec.t() * vVec);
        double uProjv = mat1x1.at<double>(0, 0);
        mat1x1 = (uVec.t() * wVec);
        double uProjw = mat1x1.at<double>(0, 0);
        mat1x1 = (vVec.t() * wVec);
        double vProjw = mat1x1.at<double>(0, 0);
        denominator = (uProjv * uProjv - ulengthsq * vlengthsq);
        double sparam = (uProjv * vProjw - vlengthsq * uProjw) / denominator;
        tparam = (uProjv * uProjw - ulengthsq * vProjw) / denominator;
#if DEBUG > 4
        ROS_ERROR_NAMED("SAR_plugin", "n = (%f,%f,%f)", n[0], n[1], n[2]);
        ROS_ERROR_NAMED("SAR_plugin", "lambda_ip = %f", tparam);
        ROS_ERROR_NAMED("SAR_plugin", "rp0 = (%f,%f,%f)",
                rp0.at<double>(0), rp0.at<double>(1), rp0.at<double>(2));
        ROS_ERROR_NAMED("SAR_plugin", "ip = (%f,%f,%f)",
                planePoint.at<double>(0), planePoint.at<double>(1), planePoint.at<double>(2));
        ROS_ERROR_NAMED("SAR_plugin", "sparam = %f tparam  = %f sparam+tparam = %f)",
                sparam, tparam, sparam + tparam);
#endif
        if (sparam >= 0.0 && sparam <= 1.0 && tparam >= 0.0 && tparam <= 1.0 && sparam + tparam < 1.0) {
            //    %ip = tp1 + sparam*uVec + tparam*vVec;
            //    %planePoint;
            //    %tp1
            //    %tp2
            //    %tp3
            ip = planePoint;
            intersect = true;
        } else {
            ip.at<double>(0, 0) = ip.at<double>(1, 0) = ip.at<double>(2, 0) = std::numeric_limits<double>::infinity();
            intersect = false;
        }
        return intersect;
    }

    GZ_REGISTER_MODEL_PLUGIN(SARPlugin);
    //GZ_REGISTER_SENSOR_PLUGIN(SARPlugin);
    //GZ_REGISTER_STATIC_SENSOR("radar", SARPlugin)
}
