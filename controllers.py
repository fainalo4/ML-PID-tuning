import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd, setpoint, umin, umax, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.umin = umin
        self.umax = umax
        self.dt= dt

    def reset(self):
        self.previous_error = 0
        self.integral = 0

    def compute(self, x):
        # Calculate error
        error = self.setpoint - x
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * self.dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out

        actual_output= max(self.umin, min(self.umax, output))
        if actual_output != output:
            # Anti-windup: reset integral term if output is saturated
            self.anti_windup(actual_output, output)
        
        # Update previous error
        self.previous_error = error
        
        return np.array(actual_output).reshape(1,)
    

    def anti_windup(self, actual_output, output):
        dts= 1
        # Adjust the integral term to prevent windup
        self.integral += (actual_output - output)*dts
        

class Relay:
    def __init__(self, setpoint, range, umin, umax):
        self.xmin= setpoint - range
        self.xmax= setpoint + range

        self.umin = umin
        self.umax = umax

        self.z = 1  # 0 for umin, 1 for umax
    
    def reset(self):
        self.z= 1

    def compute(self, x):
        
        if self.z==1 and x <= self.xmax or self.z==0 and x <= self.xmin:
            self.z= 1 
        else:
            self.z= 0
            
        if self.z == 0: 
            actual_output= self.umin
        elif self.z == 1: 
            actual_output= self.umax

        return actual_output