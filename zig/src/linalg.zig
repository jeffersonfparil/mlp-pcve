const std = @import("std");
const Allocator = std.heap.Allocator;
const Io = std.Io;
const print = std.debug.print;
const expect = std.expect;
const T = f32;

const Matrix = struct {
    data: [][]T,
    datatype: type,
    n: usize,
    p: usize,
    // const Self = @This();
    // pub fn init(io: Io, allocator: Allocator, n: usize, p: usize, datatype: type) !Matrix {
    //     return Matrix{
    //         .data = try allocator.alloc([]T, n),
    //         .type = type,
    //         .n = n,
    //         .p = p,
    //     };
    // }
};
