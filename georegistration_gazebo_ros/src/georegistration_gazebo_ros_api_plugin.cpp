/*
 * Copyright 2013 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

/* Desc: External interfaces for Gazebo
 * Author: Nate Koenig, John Hsu, Dave Coleman
 * Date: Jun 10 2013
 */

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo_config.h>
#include <georegistration_gazebo_ros/georegistration_gazebo_ros_api_plugin.h>

#include <geometry_msgs/Vector3.h>

namespace gazebo {

    GazeboRosGeoregistrationApiPlugin::GazeboRosGeoregistrationApiPlugin() :
    plugin_loaded_(false),
    stop_(false),
    //    pub_model_states_connection_count_(0),
    world_created_(false) {
        robot_namespace_.clear();
    }

    GazeboRosGeoregistrationApiPlugin::~GazeboRosGeoregistrationApiPlugin() {
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "GazeboRosGeoregistrationApiPlugin Deconstructor start");

        // Unload the sigint event
        sigint_event_.reset();
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "After sigint_event unload");

        // Don't attempt to unload this plugin if it was never loaded in the Load() function
        if (!plugin_loaded_) {
            ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Deconstructor skipped because never loaded");
            return;
        }

        // Disconnect slots
        load_gazebo_ros_api_plugin_event_.reset();
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Slots disconnected");

        //        if (pub_model_states_connection_count_ > 0) // disconnect if there are subscribers on exit
        //            pub_model_states_event_.reset();
        //        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Disconnected World Updates");

        // Stop the multi threaded ROS spinner
        async_ros_spin_->stop();
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Async ROS Spin Stopped");

        // Shutdown the ROS node
        nh_->shutdown();
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Node Handle Shutdown");

        // Shutdown ROS queue
        gazebo_callback_queue_thread_->join();
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Callback Queue Joined");

        // Delete Force and Wrench Jobs
        //        lock_.lock();
        //  for (std::vector<GazeboRosApiPlugin::WrenchBodyJob*>::iterator iter=wrench_body_jobs_.begin();iter!=wrench_body_jobs_.end();)
        //  {
        //    delete (*iter);
        //    iter = wrench_body_jobs_.erase(iter);
        //  }
        //  wrench_body_jobs_.clear();
        //        lock_.unlock();
        //  ROS_DEBUG_STREAM_NAMED("georeg_api_plugin","WrenchBodyJobs deleted");

        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Unloaded");
    }

    void GazeboRosGeoregistrationApiPlugin::shutdownSignal() {
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "shutdownSignal() recieved");
        stop_ = true;
    }

    void GazeboRosGeoregistrationApiPlugin::Load(int argc, char** argv) {
        ROS_DEBUG_STREAM_NAMED("georeg_api_plugin", "Load");

        // connect to sigint event
        sigint_event_ = gazebo::event::Events::ConnectSigInt(boost::bind(&GazeboRosGeoregistrationApiPlugin::shutdownSignal, this));

        // setup ros related
        if (!ros::isInitialized())
            ros::init(argc, argv, "georegistration_gazebo_ros", ros::init_options::NoSigintHandler);
        else
            ROS_ERROR_NAMED("georeg_api_plugin", "Something other than this gazebo_ros_api plugin started ros::init(...), command line arguments may not be parsed properly.");

        // check if the ros master is available - required
        while (!ros::master::check()) {
            ROS_WARN_STREAM_NAMED("georeg_api_plugin", "No ROS master - start roscore to continue...");
            // wait 0.5 second
            usleep(500 * 1000); // can't use ROS Time here b/c node handle is not yet initialized

            if (stop_) {
                ROS_WARN_STREAM_NAMED("georeg_api_plugin", "Canceled loading Gazebo ROS API plugin by sigint event");
                return;
            }
        }

        nh_.reset(new ros::NodeHandle("~")); // advertise topics and services in this node's namespace

        // Built-in multi-threaded ROS spinning
        async_ros_spin_.reset(new ros::AsyncSpinner(0)); // will use a thread for each CPU core
        async_ros_spin_->start();

        /// \brief setup custom callback queue
        gazebo_callback_queue_thread_.reset(new boost::thread(&GazeboRosGeoregistrationApiPlugin::gazeboQueueThread, this));

        // below needs the world to be created first
        load_gazebo_ros_api_plugin_event_ = gazebo::event::Events::ConnectWorldCreated(boost::bind(&GazeboRosGeoregistrationApiPlugin::loadGazeboRosApiPlugin, this, _1));

        plugin_loaded_ = true;
        ROS_INFO_NAMED("georeg_api_plugin", "Finished loading Gazebo ROS API Plugin.");
    }

    void GazeboRosGeoregistrationApiPlugin::loadGazeboRosApiPlugin(std::string world_name) {
        // make sure things are only called once
        lock_.lock();
        if (world_created_) {
            lock_.unlock();
            return;
        }

        // set flag to true and load this plugin
        world_created_ = true;
        lock_.unlock();

        world_ = gazebo::physics::get_world(world_name);
        if (!world_) {
            //ROS_ERROR_NAMED("api_plugin", "world name: [%s]",world->Name().c_str());
            // connect helper function to signal for scheduling torque/forces, etc
            ROS_FATAL_NAMED("georeg_api_plugin", "cannot load gazebo ros api server plugin, physics::get_world() fails to return world");
            return;
        }

        gazebonode_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
        gazebonode_->Init(world_name);
        //stat_sub_ = gazebonode_->Subscribe("~/world_stats", &GazeboRosApiPlugin::publishSimTime, this); // TODO: does not work in server plugin?
        //factory_pub_ = gazebonode_->Advertise<gazebo::msgs::Factory>("~/factory");
        //  factory_light_pub_ = gazebonode_->Advertise<gazebo::msgs::Light>("~/factory/light");
        //  light_modify_pub_ = gazebonode_->Advertise<gazebo::msgs::Light>("~/light/modify");
        //request_pub_ = gazebonode_->Advertise<gazebo::msgs::Request>("~/request");
        //response_sub_ = gazebonode_->Subscribe("~/response", &GazeboRosGeoregistrationApiPlugin::onResponse, this);

        // reset topic connection counts
        //        pub_model_states_connection_count_ = 0;

        /// \brief advertise all services
        advertiseServices();

        // hooks for applying forces, publishing simtime on /clock
        //  wrench_update_event_ = gazebo::event::Events::ConnectWorldUpdateBegin(boost::bind(&GazeboRosApiPlugin::wrenchBodySchedulerSlot,this));
    }

    void GazeboRosGeoregistrationApiPlugin::gazeboQueueThread() {
        static const double timeout = 0.001;
        while (nh_->ok()) {
            gazebo_queue_.callAvailable(ros::WallDuration(timeout));
        }
    }

    void GazeboRosGeoregistrationApiPlugin::advertiseServices() {

        // Advertise more services on the custom queue
        std::string get_modelgeom_properties_service_name("get_model_size");
        ros::AdvertiseServiceOptions get_model_size_aso =
                ros::AdvertiseServiceOptions::create<georegistration_gazebo_msgs::GetModelGeomProperties>(
                get_modelgeom_properties_service_name,
                boost::bind(&GazeboRosGeoregistrationApiPlugin::getModelProperties, this, _1, _2),
                ros::VoidPtr(), &gazebo_queue_);
        get_model_size_service_ = nh_->advertiseService(get_model_size_aso);
    }

    //    void GazeboRosGeoregistrationApiPlugin::onModelStatesConnect() {
    //        pub_model_states_connection_count_++;
    //        if (pub_model_states_connection_count_ == 1) // connect on first subscriber
    //            pub_model_states_event_ = gazebo::event::Events::ConnectWorldUpdateBegin(boost::bind(&GazeboRosGeoregistrationApiPlugin::publishModelStates, this));
    //    }
    //
    //    void GazeboRosGeoregistrationApiPlugin::onModelStatesDisconnect() {
    //        pub_model_states_connection_count_--;
    //        if (pub_model_states_connection_count_ <= 0) // disconnect with no subscribers
    //        {
    //            pub_model_states_event_.reset();
    //            if (pub_model_states_connection_count_ < 0) // should not be possible
    //                ROS_ERROR_NAMED("georeg_api_plugin", "One too mandy disconnect from pub_model_states_ in gazebo_ros.cpp? something weird");
    //        }
    //    }

    bool GazeboRosGeoregistrationApiPlugin::getModelProperties(georegistration_gazebo_msgs::GetModelGeomProperties::Request &req,
            georegistration_gazebo_msgs::GetModelGeomProperties::Response &res) {
#if GAZEBO_MAJOR_VERSION >= 8
        gazebo::physics::ModelPtr model = world_->ModelByName(req.model_name);
#else
        gazebo::physics::ModelPtr model = world_->GetModel(req.model_name);
#endif
        if (!model) {
            ROS_ERROR_NAMED("georeg_api_plugin", "GetModelProperties: model [%s] does not exist", req.model_name.c_str());
            res.success = false;
            res.status_message = "GetModelProperties: model does not exist";
            return true;
        } else {
            // get model parent name
            gazebo::physics::ModelPtr parent_model = boost::dynamic_pointer_cast<gazebo::physics::Model>(model->GetParent());
            if (parent_model) res.parent_model_name = parent_model->GetName();

            // get list of child bodies, geoms
            res.body_names.clear();
            res.geom_names.clear();
            for (unsigned int i = 0; i < model->GetChildCount(); i++) {
                gazebo::physics::LinkPtr body = boost::dynamic_pointer_cast<gazebo::physics::Link>(model->GetChild(i));
                if (body) {
                    res.body_names.push_back(body->GetName());
                    // get list of geoms
                    for (unsigned int j = 0; j < body->GetChildCount(); j++) {
                        gazebo::physics::CollisionPtr geom = boost::dynamic_pointer_cast<gazebo::physics::Collision>(body->GetChild(j));
                        if (geom) {
                            res.geom_names.push_back(geom->GetName());
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
                                res.geom_sizes.push_back(geom_size);
                                geometry_msgs::Vector3 geom_scale = geometry_msgs::Vector3();
                                geom_size.x = tmp_scale[0];
                                geom_size.y = tmp_scale[1];
                                geom_size.z = tmp_scale[2];
                                res.geom_scales.push_back(geom_scale);
                                std::cout << "Set BOX message data!" << std::endl;
                            } else if (shape->HasType(gazebo::physics::Base::PLANE_SHAPE)) {
                                gazebo::physics::PlaneShape *plane = static_cast<gazebo::physics::PlaneShape*> (shape.get());
                                ignition::math::Vector2d tmp_size2 = plane->Size();
                                //ignition::math::Vector2d tmp_scale2 = plane->Scale();
                                ignition::math::Vector3d tmp_size(tmp_size2[0], tmp_size2[1], 0);
                                ignition::math::Vector3d tmp_scale(0, 0, 0);
                                geometry_msgs::Vector3 geom_size = geometry_msgs::Vector3();
                                geom_size.x = tmp_size[0];
                                geom_size.y = tmp_size[1];
                                geom_size.z = tmp_size[2];
                                res.geom_sizes.push_back(geom_size);
                                geometry_msgs::Vector3 geom_scale = geometry_msgs::Vector3();
                                geom_size.x = tmp_scale[0];
                                geom_size.y = tmp_scale[1];
                                geom_size.z = tmp_scale[2];
                                res.geom_scales.push_back(geom_scale);
                                std::cout << "Set PLANE message data!" << std::endl;
                            }
                        }
                    }
                }
            }

            // get list of joints
            res.joint_names.clear();

            gazebo::physics::Joint_V joints = model->GetJoints();
            for (unsigned int i = 0; i < joints.size(); i++)
                res.joint_names.push_back(joints[i]->GetName());

            // get children model names
            res.child_model_names.clear();
            for (unsigned int j = 0; j < model->GetChildCount(); j++) {
                gazebo::physics::ModelPtr child_model = boost::dynamic_pointer_cast<gazebo::physics::Model>(model->GetChild(j));
                if (child_model)
                    res.child_model_names.push_back(child_model->GetName());
            }

            // is model static
            res.is_static = model->IsStatic();

            res.success = true;
            res.status_message = "GetModelProperties: got properties";
            return true;
        }
        return true;
    }

    //    void GazeboRosGeoregistrationApiPlugin::publishModelStates() {
    //        gazebo_msgs::ModelStates model_states;
    //
    //        // fill model_states
    //#if GAZEBO_MAJOR_VERSION >= 8
    //        for (unsigned int i = 0; i < world_->ModelCount(); i++) {
    //            gazebo::physics::ModelPtr model = world_->ModelByIndex(i);
    //            ignition::math::Pose3d model_pose = model->WorldPose(); // - myBody->GetCoMPose();
    //            ignition::math::Vector3d linear_vel = model->WorldLinearVel();
    //            ignition::math::Vector3d angular_vel = model->WorldAngularVel();
    //#else
    //        for (unsigned int i = 0; i < world_->GetModelCount(); i++) {
    //            gazebo::physics::ModelPtr model = world_->GetModel(i);
    //            ignition::math::Pose3d model_pose = model->GetWorldPose().Ign(); // - myBody->GetCoMPose();
    //            ignition::math::Vector3d linear_vel = model->GetWorldLinearVel().Ign();
    //            ignition::math::Vector3d angular_vel = model->GetWorldAngularVel().Ign();
    //#endif
    //            ignition::math::Vector3d pos = model_pose.Pos();
    //            ignition::math::Quaterniond rot = model_pose.Rot();
    //            geometry_msgs::Pose pose;
    //            pose.position.x = pos.X();
    //            pose.position.y = pos.Y();
    //            pose.position.z = pos.Z();
    //            pose.orientation.w = rot.W();
    //            pose.orientation.x = rot.X();
    //            pose.orientation.y = rot.Y();
    //            pose.orientation.z = rot.Z();
    //            model_states.pose.push_back(pose);
    //            model_states.name.push_back(model->GetName());
    //            geometry_msgs::Twist twist;
    //            twist.linear.x = linear_vel.X();
    //            twist.linear.y = linear_vel.Y();
    //            twist.linear.z = linear_vel.Z();
    //            twist.angular.x = angular_vel.X();
    //            twist.angular.y = angular_vel.Y();
    //            twist.angular.z = angular_vel.Z();
    //            model_states.twist.push_back(twist);
    //        }
    //        pub_model_states_.publish(model_states);
    //    }

    // Register this plugin with the simulator
    GZ_REGISTER_SYSTEM_PLUGIN(GazeboRosGeoregistrationApiPlugin)
}
