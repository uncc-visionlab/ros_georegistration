#! /usr/bin/env python
# Wrappers around the services provided by rosified gazebo

import sys
import rospy
import os
import time

from gazebo_msgs.msg import *
from gazebo_msgs.srv import *
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Wrench

def get_model_size_client(model_name, model_param_name, joint_names, joint_positions, gazebo_namespace):
    rospy.loginfo("Waiting for service %s/get_model_configuration"%gazebo_namespace)
    rospy.wait_for_service(gazebo_namespace+'/set_model_configuration')
    rospy.loginfo("temporary hack to **fix** the -J joint position option (issue #93), sleeping for 1 second to avoid race condition.");
    time.sleep(1)
    try:
      set_model_configuration = rospy.ServiceProxy(gazebo_namespace+'/set_model_configuration', SetModelConfiguration)
      rospy.loginfo("Calling service %s/set_model_configuration"%gazebo_namespace)
      resp = set_model_configuration(model_name, model_param_name, joint_names, joint_positions)
      rospy.loginfo("Set model configuration status: %s"%resp.status_message)

      return resp.success
    except rospy.ServiceException as e:
      print("Service call failed: %s" % e)

