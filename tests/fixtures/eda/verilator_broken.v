// Deliberately broken RTL, paired line-for-line with verilator_errors.txt so a
// test can assert that the quoted source line really is the offending one. The
// line numbers in that fixture are load-bearing: change this file and the
// diagnostics stop pointing at what they name.
module broken_fifo #(
    parameter WIDTH = 8
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire [WIDTH-1:0] data_in,
    output reg  [WIDTH-1:0] data_out,
    output wire [2:0]       level
);

    reg [3:0] count;

    // line 17: the RHS literal is narrower than the target
    assign data_out = 4'h0;

    // line 20: a pin the submodule does not declare
    sub_block u0 (.clk(clk), .rstn(rst_n));

    // line 23: count is 4 bits, level is 3 -- the top bit is dropped
    assign level = count;

    always @(posedge clk) begin
        if (!rst_n)
            count <= 4'd0
    end

endmodule
