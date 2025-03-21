#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf2_ros
import tf
import geometry_msgs.msg
import math

from robot_navigation.grid_manager import GridManager
from robot_navigation.path_planner import PathPlanner
from robot_navigation.robot_control import RobotController
from robot_navigation.visualization import Visualizer
from robot_navigation.debug_pose_publisher import DebugPosePublisher


VERBOSE = False


class NavigationClient:
    def __init__(self):
        rospy.init_node('navigation_client')
        
        # Get parameters
        self.debug = rospy.get_param('~debug', False)
        self.debug_map_type = rospy.get_param('~debug_map_type', 'empty')
        self.use_slam = rospy.get_param('~use_slam', False)
        
        # Initialize components
        self.grid_manager = GridManager()
        self.path_planner = PathPlanner(self.grid_manager)
        self.robot_controller = RobotController()
        self.visualizer = Visualizer()
        
        # Initialize state variables
        self.scan_points = []
        self.current_path = []
        self.goal_position = None
        self.current_pose = None
        self.last_update_pose = None
        self.path_planning_needed = False  # New flag to trigger path planning
        
        # Set up TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Initialize subscribers
        self.laser_sub = rospy.Subscriber('scan', LaserScan, self.laser_callback)
        self.goal_sub = rospy.Subscriber('move_base_simple/goal', PoseStamped, self.goal_callback)
        
        if self.use_slam:
            self.pose_sub = rospy.Subscriber('/RosAria/pose', Odometry, self.pose_callback)
            rospy.loginfo("SLAM enabled - subscribing to odometry topic")
        else:
            rospy.loginfo("SLAM disabled - using static map")
        
        if self.debug:
            # create debug pose publisher
            self.debug_pose_publisher = DebugPosePublisher(self.grid_manager)
            # Create initial debug map immediately
            self.grid_manager.create_debug_map(density=0.01, cluster_size=3, cluster_probability=0.2)
            rospy.loginfo("Initial debug map created")
            # Then start the timer for periodic updates
            # self.debug_timer = rospy.Timer(rospy.Duration(0.1), self.debug_timer_callback)

        # Add timer for path planning (2 Hz)
        self.path_planning_timer = rospy.Timer(rospy.Duration(0.5), self.path_planning_timer_callback)
        
        # Add timer for regular visualization updates (10 Hz)
        self.update_timer = rospy.Timer(rospy.Duration(0.1), self.visualization_timer_callback)
        
        # Add high-frequency timer for path execution (100 Hz)
        self.path_timer = rospy.Timer(rospy.Duration(0.01), self.path_execution_timer_callback)
    

    def visualization_timer_callback(self, event):
        """Regular timer callback for visualization updates."""
        # Update visualizations regardless of debug mode
        self.visualizer.visualize_occupancy_grid(self.grid_manager)
        self.visualizer.visualize_path(self.current_path, self.grid_manager)
        
        if not self.debug:
            self.visualizer.visualize_points(self.scan_points)

    def path_execution_timer_callback(self, event):
        """High-frequency timer callback for path execution."""
        if self.current_path:
            self.robot_controller.execute_path(self.current_path, self.grid_manager.grid_to_world)

    def laser_callback(self, scan):
        """Handle new laser scan data."""
        
        # need to update the last pose for the grid manager to do slam (transform current grid data)
        self.grid_manager.update_last_pose(self.current_pose)
            
        if not self.debug:
            self.scan_points = []
            # Clear the grid periodically to prevent stale data
            # self.grid_manager.occupancy_grid = [[False for _ in range(self.grid_manager.grid_size)] 
            #                                   for _ in range(self.grid_manager.grid_size)]
            
            for i, range_val in enumerate(scan.ranges):
                if range_val < scan.range_min or range_val > scan.range_max:
                    continue
                
                angle = scan.angle_min + (i * scan.angle_increment)
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                
                point = {
                    'x': x,
                    'y': y,
                    'distance': range_val,
                    'angle': angle
                }
                self.scan_points.append(point)
                # Make sure points are being added to grid
                self.grid_manager.update_occupancy_grid(point, self.current_pose, self.use_slam)
            
            # Add debug logging
            occupied_cells = sum(row.count(True) for row in self.grid_manager.occupancy_grid)
            rospy.logdebug(f"Grid has {occupied_cells} occupied cells")
            
            self.grid_manager.maintain_clear_zone()
            self.grid_manager.update_grid_persistence()
        
        # Update visualizations (handled in self.visualization_timer_callback(None))
        # self.visualizer.visualize_points(self.scan_points)
        # self.visualizer.visualize_occupancy_grid(self.grid_manager)
        # self.visualizer.visualize_path(self.current_path, self.grid_manager)
        
        # # Execute path if available
        # if self.current_path:
        #     self.robot_controller.execute_path(self.current_path, self.grid_manager.grid_to_world)

        self.visualization_timer_callback(None)   # update the visualizations and path planning
        
    def goal_callback(self, msg):
        """Handle new goal position."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        
        self.goal_position = self.grid_manager.world_to_grid_float(goal_x, goal_y)
        rospy.loginfo(f"New goal received at: World({goal_x:.2f}, {goal_y:.2f}) -> Grid{self.goal_position}")
        
        # Add debug info
        if self.debug:
            occupied_cells = sum(row.count(True) for row in self.grid_manager.occupancy_grid)
            rospy.loginfo(f"Current map has {occupied_cells} occupied cells")
        
        self.path_planning_needed = True
        
        # compute path
        self.current_path = self.path_planner.astar(self.goal_position)

    def transform_points_with_robot_motion(self, points, dx, dy, dyaw):
        """Transform a list of points based on robot motion.
        
        Args:
            points: List of (x, y) tuples or single (x, y) tuple
            dx, dy: Robot displacement in local frame
            dyaw: Robot rotation
            
        Returns:
            Transformed points in same format as input
        """
        # Convert to list if single point
        is_single = not isinstance(points, list)
        point_list = [points] if is_single else points
        
        transformed_points = []
        for point in point_list:
            # 1. Handle displacement
            grid_dx, grid_dy = self.grid_manager.world_to_grid_float(dx, dy)
            grid_dx = grid_dx - self.grid_manager.grid_center[0]
            grid_dy = grid_dy - self.grid_manager.grid_center[1]
            
            # Apply translation
            translated_x = point[0] - grid_dx
            translated_y = point[1] - grid_dy
            
            # 2. Handle rotation
            # Calculate relative position from robot (grid center)
            rel_x = translated_x - self.grid_manager.grid_center[0]
            rel_y = translated_y - self.grid_manager.grid_center[1]
            
            # Calculate current angle and radius
            current_angle = math.atan2(rel_y, rel_x)
            radius = math.sqrt(rel_x * rel_x + rel_y * rel_y)
            
            # Apply rotation
            new_angle = current_angle - dyaw
            
            # Convert back to grid coordinates
            new_rel_x = radius * math.cos(new_angle)
            new_rel_y = radius * math.sin(new_angle)
            
            # Convert back to absolute grid coordinates
            transformed_point = (
                new_rel_x + self.grid_manager.grid_center[0],
                new_rel_y + self.grid_manager.grid_center[1]
            )
            transformed_points.append(transformed_point)
        
        return transformed_points[0] if is_single else transformed_points

    def pose_callback(self, msg):
        """Handle new pose information."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = -euler[2] # clockwise is positive
        
        # Calculate displacement since last update
        if self.current_pose:
            # Get displacement in world frame
            dx_world = position.x - self.current_pose['x']
            dy_world = position.y - self.current_pose['y']
            dyaw = yaw - self.current_pose['yaw']
            
            # Transform displacement from world frame to robot's local frame
            cos_yaw = math.cos(-self.current_pose['yaw'])
            sin_yaw = math.sin(-self.current_pose['yaw'])
            dx = dx_world * cos_yaw - dy_world * sin_yaw
            dy = dx_world * sin_yaw + dy_world * cos_yaw
        else:
            dx = 0
            dy = 0
            dyaw = 0
        
        self.current_pose = {
            'x': position.x,
            'y': position.y,
            'yaw': yaw
        }
        
        if True:
            rospy.loginfo(f"Current pose: {self.current_pose}")
        
        if self.goal_position and (dx != 0 or dy != 0 or dyaw != 0):
            # Transform both goal and path points
            self.goal_position = self.transform_points_with_robot_motion(self.goal_position, dx, dy, dyaw)
            
            if self.current_path:
                self.current_path = self.transform_points_with_robot_motion(self.current_path, dx, dy, dyaw)
            
            if VERBOSE:
                rospy.loginfo(f"Updated goal position: {self.goal_position}")
            
            self.path_planning_needed = True
        
        self.last_update_pose = self.current_pose.copy()
        self.visualizer.visualize_robot_pose(self.grid_manager)
        
        # Update visualizations after path/grid changes
        self.visualization_timer_callback(None)
        
        self.grid_manager.maintain_clear_zone()

    def broadcast_transform(self):
        """Broadcast the transform from map to base_link."""
        if not self.current_pose:
            return

        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "base_link"
        t.child_frame_id = "odom"
        
        t.transform.translation.x = self.current_pose['x']
        t.transform.translation.y = self.current_pose['y']
        t.transform.translation.z = 0.0
        
        q = tf.transformations.quaternion_from_euler(0, 0, self.current_pose['yaw'])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)

    def path_planning_timer_callback(self, event):
        """Timer callback for path planning at 2 Hz"""
        if self.path_planning_needed and self.goal_position:
            self.current_path = self.path_planner.astar(self.goal_position)
            if not self.current_path:
                # no path found, set velocity to 0
                self.robot_controller.set_freeze_state(True)
            self.path_planning_needed = False  # Reset the flag

if __name__ == '__main__':
    try:
        navigation_client = NavigationClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass 