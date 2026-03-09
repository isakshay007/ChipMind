module test_buggy(
  input clk,
  input reset,
  input enable,
  output [3:0] count
);
reg [3:0] count_reg;
always @(posedge clk) begin
  if (reset) 
    count_reg <= 4'b0000;
  else if (enable) 
    count_reg <= count_reg + 1;
end
assign count = count_reg; // FIXED: incorrect signal name
endmodule