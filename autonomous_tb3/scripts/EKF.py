import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
import math
from tf2_ros import TransformBroadcaster

class EKFOdometry(Node):
    def __init__(self):
        super().__init__('ekf_odometry')

        self.imu_sub = self.create_subscription(String, '/combined_data', self.process_imu_data, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last_time = self.get_clock().now()
        
        # EKF Setup
        self.ekf = EKF(dim_x=5, dim_z=4)  # [x, y, theta, vx, vy]
        self.ekf.x = np.zeros(5)  # State: [x, y, theta, vx, vy]
        self.ekf.P *= 1.0  # Initial state covariance
        self.ekf.Q = np.eye(5) * 0.1  # Process noise
        self.ekf.R = np.eye(4) * 0.05  # Measurement noise

        # Encoder parameters
        self.CPR = 9000  # Encoder Counts Per Revolution
        self.R = 0.076  # Wheel radius in meters

        # Last encoder values
        self.last_encoders = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
        
        # Last IMU quaternion
        self.last_orientation = Quaternion()

    def process_imu_data(self, msg):
        try:
            parsed_data = json.loads(msg.data)
            self.get_logger().info(f"Received data: {parsed_data}")
            
            # Compute dt correctly
            current_time = self.get_clock().now()
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt <= 0:
                self.get_logger().warn("Skipping update due to non-positive dt")
                return
            
            self.last_time = current_time

            # Extract encoder values and IMU quaternion
            a, b, c, d = parsed_data.get("a", 0.0), parsed_data.get("b", 0.0), parsed_data.get("c", 0.0), parsed_data.get("d", 0.0)
            imu_q = Quaternion(
                w=parsed_data.get("w", 1.0),
                x=parsed_data.get("x", 0.0),
                y=parsed_data.get("y", 0.0),
                z=parsed_data.get("z", 0.0)
            )

            # Compute velocity using encoder differences
            vx, vy, omega = self.compute_velocity(a, b, c, d, dt)

            # Update EKF Transition Matrix (F)
            self.ekf.F = self.compute_F_matrix(dt)
            self.ekf.predict()
            
            # Use IMU quaternion directly for orientation
            self.ekf.x[0] += (vx * dt)  # x position
            self.ekf.x[1] += (vy * dt)  # y position
            self.ekf.x[2] = 0  # Keep orientation tracking in quaternion space

            # Measurement update (vx, vy, omega)
            z = np.array([vx, vy, omega, 0])  
            self.ekf.update(z, self.H_jacobian, self.measurement_function)
            
            # Store latest quaternion
            self.last_orientation = imu_q
            
            self.publish_odometry(imu_q)
            self.publish_transform(imu_q)

            # Store current encoder values for next iteration
            self.last_encoders = {"a": -a, "b": b, "c": c, "d": -d}

        except json.JSONDecodeError:
            self.get_logger().warn("Invalid JSON format received.")
        except Exception as e:
            self.get_logger().error(f"Error processing data: {e}")

    def compute_velocity(self, a, b, c, d, dt):
        """ Convert encoder values to velocity components using Mecanum kinematics. """
        a,d = -a,-d
        
        # Calculate velocity using encoder differences
        v_fl = ((a - self.last_encoders["a"]) * 2 * math.pi * self.R) / (self.CPR * dt)
        v_fr = ((b - self.last_encoders["b"]) * 2 * math.pi * self.R) / (self.CPR * dt)
        v_rl = ((c - self.last_encoders["c"]) * 2 * math.pi * self.R) / (self.CPR * dt)
        v_rr = ((d - self.last_encoders["d"]) * 2 * math.pi * self.R) / (self.CPR * dt)

        # Compute robot velocity using Mecanum wheel kinematics
        vx = (v_fl + v_fr + v_rl + v_rr) / 4  # Linear velocity in x
        vy = (-v_fl + v_fr + v_rl - v_rr) / 4  # Linear velocity in y
        omega = (-v_fl + v_fr - v_rl + v_rr) / (4 * self.R)  # Angular velocity

        return vx, vy, omega

    def compute_F_matrix(self, dt):
        """ State transition model with dt. """
        F = np.eye(5)
        F[0, 3] = dt
        F[1, 4] = dt
        return F

    def measurement_function(self, x):
        """ Measurement function mapping state to sensor measurements. """
        return np.array([x[3], x[4], x[2], x[2]])  # vx, vy, omega

    def H_jacobian(self, x):
        """ Jacobian of measurement function. """
        H = np.zeros((4, 5))
        H[0, 3] = 1  # vx
        H[1, 4] = 1  # vy
        H[2, 2] = 1  # omega
        H[3, 2] = 0  # omega should not be directly mapped
        return H

    def publish_odometry(self, imu_q):
        """ Publish odometry message. """
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"

        # Position
        odom_msg.pose.pose.position.x = self.ekf.x[0]
        odom_msg.pose.pose.position.y = self.ekf.x[1]
        odom_msg.pose.pose.orientation = imu_q  # ✅ Use IMU quaternion

        # Velocity
        odom_msg.twist.twist.linear.x = self.ekf.x[3]
        odom_msg.twist.twist.linear.y = self.ekf.x[4]

        self.odom_pub.publish(odom_msg)

    def publish_transform(self, imu_q):
        """ Publish transform from odom to base_footprint. """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"

        t.transform.translation.x = self.ekf.x[0]
        t.transform.translation.y = self.ekf.x[1]
        t.transform.rotation = imu_q  # ✅ Use IMU quaternion

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = EKFOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()