/*
 * Copyright (C) 2012-2014 Open Source Robotics Foundation
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
/*
 * Desc: External interfaces for Gazebo
 * Author: Nate Koenig, John Hsu, Dave Coleman
 * Date: 25 Apr 2010
 */

#ifndef __GEOREGISTRATION_GAZEBO_ROS_API_PLUGIN_HH__
#define __GEOREGISTRATION_GAZEBO_ROS_API_PLUGIN_HH__

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <iostream>

#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>

// ROS
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>
#include <ros/package.h>
#include <rosgraph_msgs/Clock.h>

// Services
#include "std_srvs/Empty.h"

#include "georegistration_gazebo_msgs/GetModelGeomProperties.h"
#include "georegistration_gazebo_msgs/GetModelSize.h"

// Topics
#include "georegistration_gazebo_msgs/ModelSize.h"
#include "georegistration_gazebo_msgs/ModelSizeRequest.h"
#include "georegistration_gazebo_msgs/ModelSizeResponse.h"

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Wrench.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Twist.h"

// For model pose transform to set custom joint angles
#include <ros/ros.h>
#include <boost/shared_ptr.hpp>

// For physics dynamics reconfigure
#include <dynamic_reconfigure/server.h>
#include <gazebo_ros/PhysicsConfig.h>
#include "gazebo_msgs/SetPhysicsProperties.h"
#include "gazebo_msgs/GetPhysicsProperties.h"

#include <boost/algorithm/string.hpp>

namespace gazebo {

    /// \brief A plugin loaded within the gzserver on startup.

    class GazeboRosGeoregistrationApiPlugin : public SystemPlugin {
    public:
        /// \brief Constructor
        GazeboRosGeoregistrationApiPlugin();

        /// \brief Destructor
        ~GazeboRosGeoregistrationApiPlugin();

        /// \bried Detect if sig-int shutdown signal is recieved
        void shutdownSignal();

        /// \brief Gazebo-inherited load function
        ///
        /// Called before Gazebo is loaded. Must not block.
        /// Capitalized per Gazebo cpp style guidelines
        /// \param _argc Number of command line arguments.
        /// \param _argv Array of command line arguments.
        void Load(int argc, char** argv);

        /// \brief ros queue thread for this node
        void gazeboQueueThread();

        /// \brief advertise services
        void advertiseServices();

//        /// \brief
//        void onModelStatesConnect();
//
//        /// \brief
//        void onModelStatesDisconnect();

        /// \brief
        bool getModelProperties(georegistration_gazebo_msgs::GetModelGeomProperties::Request &req,
                georegistration_gazebo_msgs::GetModelGeomProperties::Response &res);


    private:

        /// \brief
//        void publishModelStates();

        /// \brief Connect to Gazebo via its plugin interface, get a pointer to the world, start events
        void loadGazeboRosApiPlugin(std::string world_name);

        // track if the desconstructor event needs to occur
        bool plugin_loaded_;

        // detect if sigint event occurs
        bool stop_;
        gazebo::event::ConnectionPtr sigint_event_;

        std::string robot_namespace_;

        gazebo::transport::NodePtr gazebonode_;
//        gazebo::transport::SubscriberPtr stat_sub_;
//        gazebo::transport::PublisherPtr factory_pub_;
//        gazebo::transport::PublisherPtr factory_light_pub_;
//        gazebo::transport::PublisherPtr light_modify_pub_;
//        gazebo::transport::PublisherPtr request_pub_;
//        gazebo::transport::SubscriberPtr response_sub_;

        boost::shared_ptr<ros::NodeHandle> nh_;
        ros::CallbackQueue gazebo_queue_;
        boost::shared_ptr<boost::thread> gazebo_callback_queue_thread_;

        gazebo::physics::WorldPtr world_;
//        gazebo::event::ConnectionPtr pub_model_states_event_;
        gazebo::event::ConnectionPtr load_gazebo_ros_api_plugin_event_;

        ros::ServiceServer get_model_size_service_;
//        ros::Subscriber set_model_state_topic_;
//        ros::Publisher pub_model_states_;
//        int pub_model_states_connection_count_;

        // ROS comm
        boost::shared_ptr<ros::AsyncSpinner> async_ros_spin_;

        /// \brief A mutex to lock access to fields that are used in ROS message callbacks
        boost::mutex lock_;

        bool world_created_;

    };
}
#endif
