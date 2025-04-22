#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class ArucoPIDController(Node):
    def __init__(self):
        super().__init__('aruco_pid_controller')

        # PID Gains
        self.kp, self.ki, self.kd = 1.5, 0.00004, 0.004  # Distance PID
        self.kpa, self.kia, self.kda = 1.5, 0.00004, 0.004  # Yaw PID

        # Setpoints
        self.set_dist = 0.5  # Desired distance from marker (meters)
        self.set_yaw = 0.0   # Desired yaw angle (degrees)

        # Variables
        self.last_dist, self.last_yaw = 0, 0
        self.integral_dist, self.integral_yaw = 0, 0
        self.last_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds

        # Velocity Commands
        self.linear_vel, self.ang_vel = 0, 0  # Initial velocities

        # ROS2 Publishers & Subscribers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Float32, '/aruco_distance', self.distance_callback, 10)
        self.create_subscription(Float32, '/aruco_yaw', self.yaw_callback, 10)

        self.get_logger().info("✅ ArUco PID Controller Node Started")

    def distance_callback(self, msg):
        """ Callback for ArUco distance """
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
        dt = current_time - self.last_time

        if dt > 0:
            error_dist = msg.data - self.set_dist

            # PID calculations
            P = self.kp * error_dist
            self.integral_dist += error_dist * dt
            I = self.ki * self.integral_dist
            D = self.kd * (msg.data - self.last_dist) / dt

            correction = P + I + D

            # Update velocity
            self.linear_vel = max(0, min(1.0, correction))  # Limit velocity to [0, 1]

            # Update last values
            self.last_dist = msg.data
            self.last_time = current_time

            self.publish_cmd_vel()

    def yaw_callback(self, msg):
        """ Callback for ArUco yaw angle """
        current_time = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds
        dt = current_time - self.last_time

        if dt > 0:
            error_yaw = msg.data - self.set_yaw

            # PID calculations
            P_a = self.kpa * error_yaw
            self.integral_yaw += error_yaw * dt
            I_a = self.kia * self.integral_yaw
            D_a = self.kda * (msg.data - self.last_yaw) / dt

            correction_a = P_a + I_a + D_a

            # Update angular velocity
            self.ang_vel = max(-1.0, min(1.0, correction_a))  # Limit angular velocity to [-1, 1]

            # Update last values
            self.last_yaw = msg.data
            self.last_time = current_time

            self.publish_cmd_vel()

    def publish_cmd_vel(self):
        """ Publishes velocity commands to the robot """
        cmd = Twist()
        cmd.linear.x = self.linear_vel
        cmd.angular.z = self.ang_vel
        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()