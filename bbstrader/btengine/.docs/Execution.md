# Execution Handler

Execution Handler is a class hierarchy that will represent a simulated order handling mechanism and ultimately tie into a brokerage or other means of market connectivity.

The ExecutionHandler described here is exceedingly simple, since it fills all orders at the
current market price. This is highly unrealistic, but serves as a good baseline for improvement.
