ΎΘ
Ψ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ίζ

conv2d_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_73/kernel
}
$conv2d_73/kernel/Read/ReadVariableOpReadVariableOpconv2d_73/kernel*&
_output_shapes
:*
dtype0
t
conv2d_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_73/bias
m
"conv2d_73/bias/Read/ReadVariableOpReadVariableOpconv2d_73/bias*
_output_shapes
:*
dtype0
{
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	π5* 
shared_namedense_63/kernel
t
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel*
_output_shapes
:	π5*
dtype0
r
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*
shared_namedense_63/bias
k
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes
:5*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_73/kernel/m

+Adam/conv2d_73/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_73/bias/m
{
)Adam/conv2d_73/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/m*
_output_shapes
:*
dtype0

Adam/dense_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	π5*'
shared_nameAdam/dense_63/kernel/m

*Adam/dense_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/m*
_output_shapes
:	π5*
dtype0

Adam/dense_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*%
shared_nameAdam/dense_63/bias/m
y
(Adam/dense_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/m*
_output_shapes
:5*
dtype0

Adam/conv2d_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_73/kernel/v

+Adam/conv2d_73/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_73/bias/v
{
)Adam/conv2d_73/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/v*
_output_shapes
:*
dtype0

Adam/dense_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	π5*'
shared_nameAdam/dense_63/kernel/v

*Adam/dense_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/kernel/v*
_output_shapes
:	π5*
dtype0

Adam/dense_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:5*%
shared_nameAdam/dense_63/bias/v
y
(Adam/dense_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_63/bias/v*
_output_shapes
:5*
dtype0

NoOpNoOp
Ά&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ρ%
valueη%Bδ% Bέ%
§
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*

#iter

$beta_1

%beta_2
	&decay
'learning_ratemHmImJmKvLvMvNvO*
 
0
1
2
3*
 
0
1
2
3*
* 
°
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

-serving_default* 
`Z
VARIABLE_VALUEconv2d_73/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_73/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_63/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_63/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

=0
>1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	?total
	@count
A	variables
B	keras_api*
H
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

A	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

C0
D1*

F	variables*
}
VARIABLE_VALUEAdam/conv2d_73/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_73/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_63/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_63/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_73/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_73/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_63/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_63/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_conv2d_73_inputPlaceholder*/
_output_shapes
:?????????2/*
dtype0*$
shape:?????????2/

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_73_inputconv2d_73/kernelconv2d_73/biasdense_63/kerneldense_63/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_313246
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ί
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_73/kernel/Read/ReadVariableOp"conv2d_73/bias/Read/ReadVariableOp#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_73/kernel/m/Read/ReadVariableOp)Adam/conv2d_73/bias/m/Read/ReadVariableOp*Adam/dense_63/kernel/m/Read/ReadVariableOp(Adam/dense_63/bias/m/Read/ReadVariableOp+Adam/conv2d_73/kernel/v/Read/ReadVariableOp)Adam/conv2d_73/bias/v/Read/ReadVariableOp*Adam/dense_63/kernel/v/Read/ReadVariableOp(Adam/dense_63/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_313383

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_73/kernelconv2d_73/biasdense_63/kerneldense_63/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_73/kernel/mAdam/conv2d_73/bias/mAdam/dense_63/kernel/mAdam/dense_63/bias/mAdam/conv2d_73/kernel/vAdam/conv2d_73/bias/vAdam/dense_63/kernel/vAdam/dense_63/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_313456η

ώ
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313266

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????0-i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0-w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
ο

*__inference_conv2d_73_layer_call_fn_313255

inputs!
unknown:
	unknown_0:
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????0-`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2/: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
―
Ύ
I__inference_sequential_36_layer_call_and_return_conditional_losses_313038

inputs*
conv2d_73_313007:
conv2d_73_313009:"
dense_63_313032:	π5
dense_63_313034:5
identity’!conv2d_73/StatefulPartitionedCall’ dense_63/StatefulPartitionedCall?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_73_313007conv2d_73_313009*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006δ
flatten_36/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_36/PartitionedCall:output:0dense_63_313032dense_63_313034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_63_layer_call_and_return_conditional_losses_313031x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5
NoOpNoOp"^conv2d_73/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
±U
»
"__inference__traced_restore_313456
file_prefix;
!assignvariableop_conv2d_73_kernel:/
!assignvariableop_1_conv2d_73_bias:5
"assignvariableop_2_dense_63_kernel:	π5.
 assignvariableop_3_dense_63_bias:5&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: E
+assignvariableop_13_adam_conv2d_73_kernel_m:7
)assignvariableop_14_adam_conv2d_73_bias_m:=
*assignvariableop_15_adam_dense_63_kernel_m:	π56
(assignvariableop_16_adam_dense_63_bias_m:5E
+assignvariableop_17_adam_conv2d_73_kernel_v:7
)assignvariableop_18_adam_conv2d_73_bias_v:=
*assignvariableop_19_adam_dense_63_kernel_v:	π56
(assignvariableop_20_adam_dense_63_bias_v:5
identity_22’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ύ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*δ

valueΪ
BΧ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_73_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_73_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_63_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_63_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp+assignvariableop_13_adam_conv2d_73_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_conv2d_73_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_63_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_63_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_73_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_73_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_63_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_63_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
τ
α
I__inference_sequential_36_layer_call_and_return_conditional_losses_313211

inputsB
(conv2d_73_conv2d_readvariableop_resource:7
)conv2d_73_biasadd_readvariableop_resource::
'dense_63_matmul_readvariableop_resource:	π56
(dense_63_biasadd_readvariableop_resource:5
identity’ conv2d_73/BiasAdd/ReadVariableOp’conv2d_73/Conv2D/ReadVariableOp’dense_63/BiasAdd/ReadVariableOp’dense_63/MatMul/ReadVariableOp
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_73/Conv2DConv2Dinputs'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-*
paddingVALID*
strides

 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0-a
flatten_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"????p  
flatten_36/ReshapeReshapeconv2d_73/Relu:activations:0flatten_36/Const:output:0*
T0*(
_output_shapes
:?????????π
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes
:	π5*
dtype0
dense_63/MatMulMatMulflatten_36/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5h
dense_63/SoftmaxSoftmaxdense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????5i
IdentityIdentitydense_63/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????5Ξ
NoOpNoOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
€

φ
D__inference_dense_63_layer_call_and_return_conditional_losses_313031

inputs1
matmul_readvariableop_resource:	π5-
biasadd_readvariableop_resource:5
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	π5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????5`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????π
 
_user_specified_nameinputs
Θ
b
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????p  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????πY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0-:W S
/
_output_shapes
:?????????0-
 
_user_specified_nameinputs

²
!__inference__wrapped_model_312988
conv2d_73_inputP
6sequential_36_conv2d_73_conv2d_readvariableop_resource:E
7sequential_36_conv2d_73_biasadd_readvariableop_resource:H
5sequential_36_dense_63_matmul_readvariableop_resource:	π5D
6sequential_36_dense_63_biasadd_readvariableop_resource:5
identity’.sequential_36/conv2d_73/BiasAdd/ReadVariableOp’-sequential_36/conv2d_73/Conv2D/ReadVariableOp’-sequential_36/dense_63/BiasAdd/ReadVariableOp’,sequential_36/dense_63/MatMul/ReadVariableOp¬
-sequential_36/conv2d_73/Conv2D/ReadVariableOpReadVariableOp6sequential_36_conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Σ
sequential_36/conv2d_73/Conv2DConv2Dconv2d_73_input5sequential_36/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-*
paddingVALID*
strides
’
.sequential_36/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp7sequential_36_conv2d_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ε
sequential_36/conv2d_73/BiasAddBiasAdd'sequential_36/conv2d_73/Conv2D:output:06sequential_36/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-
sequential_36/conv2d_73/ReluRelu(sequential_36/conv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0-o
sequential_36/flatten_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"????p  ³
 sequential_36/flatten_36/ReshapeReshape*sequential_36/conv2d_73/Relu:activations:0'sequential_36/flatten_36/Const:output:0*
T0*(
_output_shapes
:?????????π£
,sequential_36/dense_63/MatMul/ReadVariableOpReadVariableOp5sequential_36_dense_63_matmul_readvariableop_resource*
_output_shapes
:	π5*
dtype0Ί
sequential_36/dense_63/MatMulMatMul)sequential_36/flatten_36/Reshape:output:04sequential_36/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5 
-sequential_36/dense_63/BiasAdd/ReadVariableOpReadVariableOp6sequential_36_dense_63_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0»
sequential_36/dense_63/BiasAddBiasAdd'sequential_36/dense_63/MatMul:product:05sequential_36/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5
sequential_36/dense_63/SoftmaxSoftmax'sequential_36/dense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????5w
IdentityIdentity(sequential_36/dense_63/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????5
NoOpNoOp/^sequential_36/conv2d_73/BiasAdd/ReadVariableOp.^sequential_36/conv2d_73/Conv2D/ReadVariableOp.^sequential_36/dense_63/BiasAdd/ReadVariableOp-^sequential_36/dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2`
.sequential_36/conv2d_73/BiasAdd/ReadVariableOp.sequential_36/conv2d_73/BiasAdd/ReadVariableOp2^
-sequential_36/conv2d_73/Conv2D/ReadVariableOp-sequential_36/conv2d_73/Conv2D/ReadVariableOp2^
-sequential_36/dense_63/BiasAdd/ReadVariableOp-sequential_36/dense_63/BiasAdd/ReadVariableOp2\
,sequential_36/dense_63/MatMul/ReadVariableOp,sequential_36/dense_63/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input
Ά
G
+__inference_flatten_36_layer_call_fn_313271

inputs
identity΅
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0-:W S
/
_output_shapes
:?????????0-
 
_user_specified_nameinputs
―
Ύ
I__inference_sequential_36_layer_call_and_return_conditional_losses_313105

inputs*
conv2d_73_313093:
conv2d_73_313095:"
dense_63_313099:	π5
dense_63_313101:5
identity’!conv2d_73/StatefulPartitionedCall’ dense_63/StatefulPartitionedCall?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_73_313093conv2d_73_313095*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006δ
flatten_36/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_36/PartitionedCall:output:0dense_63_313099dense_63_313101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_63_layer_call_and_return_conditional_losses_313031x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5
NoOpNoOp"^conv2d_73/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs

ώ
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????0-i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????0-w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????2/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
Θ

)__inference_dense_63_layer_call_fn_313286

inputs
unknown:	π5
	unknown_0:5
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_63_layer_call_and_return_conditional_losses_313031o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????π: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????π
 
_user_specified_nameinputs
τ
α
I__inference_sequential_36_layer_call_and_return_conditional_losses_313231

inputsB
(conv2d_73_conv2d_readvariableop_resource:7
)conv2d_73_biasadd_readvariableop_resource::
'dense_63_matmul_readvariableop_resource:	π56
(dense_63_biasadd_readvariableop_resource:5
identity’ conv2d_73/BiasAdd/ReadVariableOp’conv2d_73/Conv2D/ReadVariableOp’dense_63/BiasAdd/ReadVariableOp’dense_63/MatMul/ReadVariableOp
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_73/Conv2DConv2Dinputs'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-*
paddingVALID*
strides

 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????0-l
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*/
_output_shapes
:?????????0-a
flatten_36/ConstConst*
_output_shapes
:*
dtype0*
valueB"????p  
flatten_36/ReshapeReshapeconv2d_73/Relu:activations:0flatten_36/Const:output:0*
T0*(
_output_shapes
:?????????π
dense_63/MatMul/ReadVariableOpReadVariableOp'dense_63_matmul_readvariableop_resource*
_output_shapes
:	π5*
dtype0
dense_63/MatMulMatMulflatten_36/Reshape:output:0&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5
dense_63/BiasAdd/ReadVariableOpReadVariableOp(dense_63_biasadd_readvariableop_resource*
_output_shapes
:5*
dtype0
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5h
dense_63/SoftmaxSoftmaxdense_63/BiasAdd:output:0*
T0*'
_output_shapes
:?????????5i
IdentityIdentitydense_63/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????5Ξ
NoOpNoOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
Χ
γ
.__inference_sequential_36_layer_call_fn_313049
conv2d_73_input!
unknown:
	unknown_0:
	unknown_1:	π5
	unknown_2:5
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_73_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313038o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input
€

φ
D__inference_dense_63_layer_call_and_return_conditional_losses_313297

inputs1
matmul_readvariableop_resource:	π5-
biasadd_readvariableop_resource:5
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	π5*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:5*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????5V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????5`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????5w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????π: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????π
 
_user_specified_nameinputs
Θ
b
F__inference_flatten_36_layer_call_and_return_conditional_losses_313277

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????p  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????πY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????π"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????0-:W S
/
_output_shapes
:?????????0-
 
_user_specified_nameinputs
Ό
Ϊ
.__inference_sequential_36_layer_call_fn_313191

inputs!
unknown:
	unknown_0:
	unknown_1:	π5
	unknown_2:5
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
τ1
Ϊ
__inference__traced_save_313383
file_prefix/
+savev2_conv2d_73_kernel_read_readvariableop-
)savev2_conv2d_73_bias_read_readvariableop.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_73_kernel_m_read_readvariableop4
0savev2_adam_conv2d_73_bias_m_read_readvariableop5
1savev2_adam_dense_63_kernel_m_read_readvariableop3
/savev2_adam_dense_63_bias_m_read_readvariableop6
2savev2_adam_conv2d_73_kernel_v_read_readvariableop4
0savev2_adam_conv2d_73_bias_v_read_readvariableop5
1savev2_adam_dense_63_kernel_v_read_readvariableop3
/savev2_adam_dense_63_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: »
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*δ

valueΪ
BΧ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B ί
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_73_kernel_read_readvariableop)savev2_conv2d_73_bias_read_readvariableop*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_73_kernel_m_read_readvariableop0savev2_adam_conv2d_73_bias_m_read_readvariableop1savev2_adam_dense_63_kernel_m_read_readvariableop/savev2_adam_dense_63_bias_m_read_readvariableop2savev2_adam_conv2d_73_kernel_v_read_readvariableop0savev2_adam_conv2d_73_bias_v_read_readvariableop1savev2_adam_dense_63_kernel_v_read_readvariableop/savev2_adam_dense_63_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*¦
_input_shapes
: :::	π5:5: : : : : : : : : :::	π5:5:::	π5:5: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	π5: 

_output_shapes
:5:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	π5: 

_output_shapes
:5:,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	π5: 

_output_shapes
:5:

_output_shapes
: 
Χ
γ
.__inference_sequential_36_layer_call_fn_313129
conv2d_73_input!
unknown:
	unknown_0:
	unknown_1:	π5
	unknown_2:5
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_73_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313105o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input
₯
Ω
$__inference_signature_wrapper_313246
conv2d_73_input!
unknown:
	unknown_0:
	unknown_1:	π5
	unknown_2:5
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallconv2d_73_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_312988o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input
Κ
Η
I__inference_sequential_36_layer_call_and_return_conditional_losses_313159
conv2d_73_input*
conv2d_73_313147:
conv2d_73_313149:"
dense_63_313153:	π5
dense_63_313155:5
identity’!conv2d_73/StatefulPartitionedCall’ dense_63/StatefulPartitionedCall
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCallconv2d_73_inputconv2d_73_313147conv2d_73_313149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006δ
flatten_36/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_36/PartitionedCall:output:0dense_63_313153dense_63_313155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_63_layer_call_and_return_conditional_losses_313031x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5
NoOpNoOp"^conv2d_73/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input
Ό
Ϊ
.__inference_sequential_36_layer_call_fn_313178

inputs!
unknown:
	unknown_0:
	unknown_1:	π5
	unknown_2:5
identity’StatefulPartitionedCallϋ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_36_layer_call_and_return_conditional_losses_313038o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????2/
 
_user_specified_nameinputs
Κ
Η
I__inference_sequential_36_layer_call_and_return_conditional_losses_313144
conv2d_73_input*
conv2d_73_313132:
conv2d_73_313134:"
dense_63_313138:	π5
dense_63_313140:5
identity’!conv2d_73/StatefulPartitionedCall’ dense_63/StatefulPartitionedCall
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCallconv2d_73_inputconv2d_73_313132conv2d_73_313134*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????0-*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313006δ
flatten_36/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????π* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_36_layer_call_and_return_conditional_losses_313018
 dense_63/StatefulPartitionedCallStatefulPartitionedCall#flatten_36/PartitionedCall:output:0dense_63_313138dense_63_313140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_63_layer_call_and_return_conditional_losses_313031x
IdentityIdentity)dense_63/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????5
NoOpNoOp"^conv2d_73/StatefulPartitionedCall!^dense_63/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????2/: : : : 2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall:` \
/
_output_shapes
:?????????2/
)
_user_specified_nameconv2d_73_input"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Γ
serving_default―
S
conv2d_73_input@
!serving_default_conv2d_73_input:0?????????2/<
dense_630
StatefulPartitionedCall:0?????????5tensorflow/serving/predict:ώM
Α
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer

#iter

$beta_1

%beta_2
	&decay
'learning_ratemHmImJmKvLvMvNvO"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
(non_trainable_variables

)layers
*metrics
+layer_regularization_losses
,layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_36_layer_call_fn_313049
.__inference_sequential_36_layer_call_fn_313178
.__inference_sequential_36_layer_call_fn_313191
.__inference_sequential_36_layer_call_fn_313129ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
I__inference_sequential_36_layer_call_and_return_conditional_losses_313211
I__inference_sequential_36_layer_call_and_return_conditional_losses_313231
I__inference_sequential_36_layer_call_and_return_conditional_losses_313144
I__inference_sequential_36_layer_call_and_return_conditional_losses_313159ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΤBΡ
!__inference__wrapped_model_312988conv2d_73_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
-serving_default"
signature_map
*:(2conv2d_73/kernel
:2conv2d_73/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Τ2Ρ
*__inference_conv2d_73_layer_call_fn_313255’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ο2μ
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313266’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Υ2?
+__inference_flatten_36_layer_call_fn_313271’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_flatten_36_layer_call_and_return_conditional_losses_313277’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
": 	π52dense_63/kernel
:52dense_63/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Σ2Π
)__inference_dense_63_layer_call_fn_313286’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ξ2λ
D__inference_dense_63_layer_call_and_return_conditional_losses_313297’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΣBΠ
$__inference_signature_wrapper_313246conv2d_73_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	?total
	@count
A	variables
B	keras_api"
_tf_keras_metric
^
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
/:-2Adam/conv2d_73/kernel/m
!:2Adam/conv2d_73/bias/m
':%	π52Adam/dense_63/kernel/m
 :52Adam/dense_63/bias/m
/:-2Adam/conv2d_73/kernel/v
!:2Adam/conv2d_73/bias/v
':%	π52Adam/dense_63/kernel/v
 :52Adam/dense_63/bias/v’
!__inference__wrapped_model_312988}@’=
6’3
1.
conv2d_73_input?????????2/
ͺ "3ͺ0
.
dense_63"
dense_63?????????5΅
E__inference_conv2d_73_layer_call_and_return_conditional_losses_313266l7’4
-’*
(%
inputs?????????2/
ͺ "-’*
# 
0?????????0-
 
*__inference_conv2d_73_layer_call_fn_313255_7’4
-’*
(%
inputs?????????2/
ͺ " ?????????0-₯
D__inference_dense_63_layer_call_and_return_conditional_losses_313297]0’-
&’#
!
inputs?????????π
ͺ "%’"

0?????????5
 }
)__inference_dense_63_layer_call_fn_313286P0’-
&’#
!
inputs?????????π
ͺ "?????????5«
F__inference_flatten_36_layer_call_and_return_conditional_losses_313277a7’4
-’*
(%
inputs?????????0-
ͺ "&’#

0?????????π
 
+__inference_flatten_36_layer_call_fn_313271T7’4
-’*
(%
inputs?????????0-
ͺ "?????????πΔ
I__inference_sequential_36_layer_call_and_return_conditional_losses_313144wH’E
>’;
1.
conv2d_73_input?????????2/
p 

 
ͺ "%’"

0?????????5
 Δ
I__inference_sequential_36_layer_call_and_return_conditional_losses_313159wH’E
>’;
1.
conv2d_73_input?????????2/
p

 
ͺ "%’"

0?????????5
 »
I__inference_sequential_36_layer_call_and_return_conditional_losses_313211n?’<
5’2
(%
inputs?????????2/
p 

 
ͺ "%’"

0?????????5
 »
I__inference_sequential_36_layer_call_and_return_conditional_losses_313231n?’<
5’2
(%
inputs?????????2/
p

 
ͺ "%’"

0?????????5
 
.__inference_sequential_36_layer_call_fn_313049jH’E
>’;
1.
conv2d_73_input?????????2/
p 

 
ͺ "?????????5
.__inference_sequential_36_layer_call_fn_313129jH’E
>’;
1.
conv2d_73_input?????????2/
p

 
ͺ "?????????5
.__inference_sequential_36_layer_call_fn_313178a?’<
5’2
(%
inputs?????????2/
p 

 
ͺ "?????????5
.__inference_sequential_36_layer_call_fn_313191a?’<
5’2
(%
inputs?????????2/
p

 
ͺ "?????????5Ή
$__inference_signature_wrapper_313246S’P
’ 
IͺF
D
conv2d_73_input1.
conv2d_73_input?????????2/"3ͺ0
.
dense_63"
dense_63?????????5