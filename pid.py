class Controller:
    def __init__(self, Kp, Ki, Kd, setpoint, umin, umax):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.umin = umin
        self.umax = umax

    def compute(self, process_variable, dt):
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out

        actual_output= max(self.umin, min(self.umax, output))

        if actual_output != output:
            # Anti-windup: reset integral term if output is saturated
            self.integral = 0
        
        # Update previous error
        self.previous_error = error
        
        return actual_output
    