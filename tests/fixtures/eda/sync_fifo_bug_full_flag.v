// The SAME FIFO with one seeded bug: `full` asserts an entry too late, so the
// FIFO accepts a ninth write into an eight-deep memory and silently corrupts it.
// This is the fixture the fix loop has to converge on.
module sync_fifo #(
    parameter WIDTH = 8,
    parameter DEPTH = 8
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   wr_en,
    input  wire                   rd_en,
    input  wire [WIDTH-1:0]       data_in,
    output reg  [WIDTH-1:0]       data_out,
    output wire                   full,
    output wire                   empty,
    output reg  [$clog2(DEPTH):0] count
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH)-1:0] wr_ptr;
    reg [$clog2(DEPTH)-1:0] rd_ptr;

    wire do_write = wr_en && !full;
    wire do_read  = rd_en && !empty;

    assign full  = (count == DEPTH + 1);   // BUG: should be (count == DEPTH)
    assign empty = (count == 0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr   <= 0;
            rd_ptr   <= 0;
            count    <= 0;
            data_out <= 0;
        end else begin
            if (do_write) begin
                mem[wr_ptr] <= data_in;
                wr_ptr      <= wr_ptr + 1'b1;
            end
            if (do_read) begin
                data_out <= mem[rd_ptr];
                rd_ptr   <= rd_ptr + 1'b1;
            end
            case ({do_write, do_read})
                2'b10:   count <= count + 1'b1;
                2'b01:   count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
