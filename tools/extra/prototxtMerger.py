print "[EXECUTING]"

import os
import sys
import argparse

#CMD Arguments
description = ('Merge two caffe prototxt files.')
parser = argparse.ArgumentParser(description=description)
parser.add_argument('prototxt_path_A',
                    help='Path to first input prototxt file (will be overwritten to resolve conflict).')
parser.add_argument('prototxt_path_B',
                    help='Path to the second input prototxt file (will overwrite to resolve conflict).')
parser.add_argument('prototxt_path_out',
                    help='Path to the output prototxt file.')
parser.add_argument('proto_type',
                    help='Options: \'net\', \'solver\'')
parser.add_argument("--show", help="print output prototxt on screen",
                    action="store_true")

args = parser.parse_args()
#args = parser.parse_args([r"D:\Users\Amogh\Projects\PL2Workspace\PIRESPP\global_net.prototxt",
#                          r"D:\Users\Amogh\Projects\PL2Workspace\PIRESPP\Cheekbones\local_net.prototxt",
#                          r"D:\Users\Amogh\Projects\PL2Workspace\PIRESPP\Cheekbones\net.prototxt",
#                          'net'])


from google.protobuf import text_format as tf
sys.path.insert(0, r"D:\Users\Amogh\Projects\PL2Workspace\Caffe-Repository\caffe\Build\x64\Release\pycaffe")
from caffe.proto import caffe_pb2 as pb2

inputfileA = open(args.prototxt_path_A, 'r')
inputfileB = open(args.prototxt_path_B, 'r')

if args.proto_type == "solver":
    paramA = pb2.SolverParameter()
    paramB = pb2.SolverParameter()
elif args.proto_type == "net":
    paramA = pb2.NetParameter()
    paramB = pb2.NetParameter()
else:
    print "[FATAL ERROR: unrecognized proto_type.]"
    quit()

tf.Merge(str(inputfileA.read()), paramA)
tf.Merge(str(inputfileB.read()), paramB)

if args.proto_type == "net":
    for iA in range(len(paramA.layer)):
        iB = 0;
        while iB < len(paramB.layer):
            if paramA.layer[iA].name == paramB.layer[iB].name:
                paramA.layer[iA].MergeFrom(paramB.layer[iB])
                del paramB.layer[iB]
                print "[ Merged layer:", paramA.layer[iA].name, "]"
                print tf.MessageToString(paramA.layer[iA], use_index_order = True)
                print "---"
            else: iB += 1

paramA.MergeFrom(paramB) #Do this for both solver and net

outputstr = tf.MessageToString(paramA, use_index_order = True)

if args.show:
    print "[Output prototxt]"
    print outputstr
    print "---"

outputfile = open(args.prototxt_path_out, "w")
outputfile.write(outputstr)
outputfile.close()

print "[ Output prototxt saved to", args.prototxt_path_out, "]"

print "[DONE]" 