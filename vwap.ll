; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define void @qfx_kernel(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14) {
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, ptr %1, 1
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %2, 2
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %3, 3, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %4, 4, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %6, 1
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %7, 2
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, i64 %8, 3, 0
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 %9, 4, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %10, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, ptr %11, 1
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %12, 2
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 %13, 3, 0
  %30 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, i64 %14, 4, 0
  %31 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %32 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %31, 0
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %32, ptr %31, 1
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, i64 0, 2
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 1024, 3, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 1, 4, 0
  br label %37

37:                                               ; preds = %40, %15
  %38 = phi i64 [ %47, %40 ], [ 0, %15 ]
  %39 = icmp slt i64 %38, 1024
  br i1 %39, label %40, label %48

40:                                               ; preds = %37
  %41 = getelementptr float, ptr %1, i64 %38
  %42 = load float, ptr %41, align 4
  %43 = getelementptr float, ptr %6, i64 %38
  %44 = load float, ptr %43, align 4
  %45 = fmul float %42, %44
  %46 = getelementptr float, ptr %31, i64 %38
  store float %45, ptr %46, align 4
  %47 = add i64 %38, 1
  br label %37

48:                                               ; preds = %37
  %49 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %49, 0
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, ptr %49, 1
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, i64 0, 2
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, i64 1024, 3, 0
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, i64 1, 4, 0
  br label %55

55:                                               ; preds = %79, %48
  %56 = phi i64 [ %81, %79 ], [ 0, %48 ]
  %57 = icmp slt i64 %56, 1024
  br i1 %57, label %58, label %82

58:                                               ; preds = %55
  br label %59

59:                                               ; preds = %76, %58
  %60 = phi i64 [ %78, %76 ], [ 0, %58 ]
  %61 = phi float [ %77, %76 ], [ 0.000000e+00, %58 ]
  %62 = icmp slt i64 %60, 20
  br i1 %62, label %63, label %79

63:                                               ; preds = %59
  %64 = add i64 %56, 9
  %65 = sub i64 %64, %60
  %66 = icmp sge i64 %65, 0
  %67 = icmp slt i64 %65, 1024
  %68 = and i1 %66, %67
  br i1 %68, label %69, label %73

69:                                               ; preds = %63
  %70 = getelementptr float, ptr %31, i64 %65
  %71 = load float, ptr %70, align 4
  %72 = fmul float %71, 0x3FA99999A0000000
  br label %74

73:                                               ; preds = %63
  br label %74

74:                                               ; preds = %69, %73
  %75 = phi float [ 0.000000e+00, %73 ], [ %72, %69 ]
  br label %76

76:                                               ; preds = %74
  %77 = fadd float %61, %75
  %78 = add i64 %60, 1
  br label %59

79:                                               ; preds = %59
  %80 = getelementptr float, ptr %49, i64 %56
  store float %61, ptr %80, align 4
  %81 = add i64 %56, 1
  br label %55

82:                                               ; preds = %55
  %83 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %84 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %83, 0
  %85 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %84, ptr %83, 1
  %86 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %85, i64 0, 2
  %87 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %86, i64 1024, 3, 0
  %88 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %87, i64 1, 4, 0
  br label %89

89:                                               ; preds = %113, %82
  %90 = phi i64 [ %115, %113 ], [ 0, %82 ]
  %91 = icmp slt i64 %90, 1024
  br i1 %91, label %92, label %116

92:                                               ; preds = %89
  br label %93

93:                                               ; preds = %110, %92
  %94 = phi i64 [ %112, %110 ], [ 0, %92 ]
  %95 = phi float [ %111, %110 ], [ 0.000000e+00, %92 ]
  %96 = icmp slt i64 %94, 20
  br i1 %96, label %97, label %113

97:                                               ; preds = %93
  %98 = add i64 %90, 9
  %99 = sub i64 %98, %94
  %100 = icmp sge i64 %99, 0
  %101 = icmp slt i64 %99, 1024
  %102 = and i1 %100, %101
  br i1 %102, label %103, label %107

103:                                              ; preds = %97
  %104 = getelementptr float, ptr %6, i64 %99
  %105 = load float, ptr %104, align 4
  %106 = fmul float %105, 0x3FA99999A0000000
  br label %108

107:                                              ; preds = %97
  br label %108

108:                                              ; preds = %103, %107
  %109 = phi float [ 0.000000e+00, %107 ], [ %106, %103 ]
  br label %110

110:                                              ; preds = %108
  %111 = fadd float %95, %109
  %112 = add i64 %94, 1
  br label %93

113:                                              ; preds = %93
  %114 = getelementptr float, ptr %83, i64 %90
  store float %95, ptr %114, align 4
  %115 = add i64 %90, 1
  br label %89

116:                                              ; preds = %89
  %117 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %118 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %117, 0
  %119 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %118, ptr %117, 1
  %120 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %119, i64 0, 2
  %121 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %120, i64 1024, 3, 0
  %122 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %121, i64 1, 4, 0
  br label %123

123:                                              ; preds = %126, %116
  %124 = phi i64 [ %135, %126 ], [ 0, %116 ]
  %125 = icmp slt i64 %124, 1024
  br i1 %125, label %126, label %136

126:                                              ; preds = %123
  %127 = getelementptr float, ptr %49, i64 %124
  %128 = load float, ptr %127, align 4
  %129 = getelementptr float, ptr %83, i64 %124
  %130 = load float, ptr %129, align 4
  %131 = fcmp olt float %130, 0x3E45798EE0000000
  %132 = select i1 %131, float 0x3E45798EE0000000, float %130
  %133 = fdiv float %128, %132
  %134 = getelementptr float, ptr %117, i64 %124
  store float %133, ptr %134, align 4
  %135 = add i64 %124, 1
  br label %123

136:                                              ; preds = %123
  %137 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %138 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %137, 0
  %139 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %138, ptr %137, 1
  %140 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %139, i64 0, 2
  %141 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %140, i64 1024, 3, 0
  %142 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %141, i64 1, 4, 0
  br label %143

143:                                              ; preds = %167, %136
  %144 = phi i64 [ %169, %167 ], [ 0, %136 ]
  %145 = icmp slt i64 %144, 1024
  br i1 %145, label %146, label %170

146:                                              ; preds = %143
  br label %147

147:                                              ; preds = %164, %146
  %148 = phi i64 [ %166, %164 ], [ 0, %146 ]
  %149 = phi float [ %165, %164 ], [ 0.000000e+00, %146 ]
  %150 = icmp slt i64 %148, 20
  br i1 %150, label %151, label %167

151:                                              ; preds = %147
  %152 = add i64 %144, 9
  %153 = sub i64 %152, %148
  %154 = icmp sge i64 %153, 0
  %155 = icmp slt i64 %153, 1024
  %156 = and i1 %154, %155
  br i1 %156, label %157, label %161

157:                                              ; preds = %151
  %158 = getelementptr float, ptr %1, i64 %153
  %159 = load float, ptr %158, align 4
  %160 = fmul float %159, 0x3FA99999A0000000
  br label %162

161:                                              ; preds = %151
  br label %162

162:                                              ; preds = %157, %161
  %163 = phi float [ 0.000000e+00, %161 ], [ %160, %157 ]
  br label %164

164:                                              ; preds = %162
  %165 = fadd float %149, %163
  %166 = add i64 %148, 1
  br label %147

167:                                              ; preds = %147
  %168 = getelementptr float, ptr %137, i64 %144
  store float %149, ptr %168, align 4
  %169 = add i64 %144, 1
  br label %143

170:                                              ; preds = %143
  %171 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %172 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %171, 0
  %173 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %172, ptr %171, 1
  %174 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %173, i64 0, 2
  %175 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %174, i64 1024, 3, 0
  %176 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %175, i64 1, 4, 0
  br label %177

177:                                              ; preds = %180, %170
  %178 = phi i64 [ %185, %180 ], [ 0, %170 ]
  %179 = icmp slt i64 %178, 1024
  br i1 %179, label %180, label %186

180:                                              ; preds = %177
  %181 = getelementptr float, ptr %1, i64 %178
  %182 = load float, ptr %181, align 4
  %183 = fmul float %182, %182
  %184 = getelementptr float, ptr %171, i64 %178
  store float %183, ptr %184, align 4
  %185 = add i64 %178, 1
  br label %177

186:                                              ; preds = %177
  %187 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %188 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %187, 0
  %189 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %188, ptr %187, 1
  %190 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %189, i64 0, 2
  %191 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %190, i64 1024, 3, 0
  %192 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %191, i64 1, 4, 0
  br label %193

193:                                              ; preds = %217, %186
  %194 = phi i64 [ %219, %217 ], [ 0, %186 ]
  %195 = icmp slt i64 %194, 1024
  br i1 %195, label %196, label %220

196:                                              ; preds = %193
  br label %197

197:                                              ; preds = %214, %196
  %198 = phi i64 [ %216, %214 ], [ 0, %196 ]
  %199 = phi float [ %215, %214 ], [ 0.000000e+00, %196 ]
  %200 = icmp slt i64 %198, 20
  br i1 %200, label %201, label %217

201:                                              ; preds = %197
  %202 = add i64 %194, 9
  %203 = sub i64 %202, %198
  %204 = icmp sge i64 %203, 0
  %205 = icmp slt i64 %203, 1024
  %206 = and i1 %204, %205
  br i1 %206, label %207, label %211

207:                                              ; preds = %201
  %208 = getelementptr float, ptr %171, i64 %203
  %209 = load float, ptr %208, align 4
  %210 = fmul float %209, 0x3FA99999A0000000
  br label %212

211:                                              ; preds = %201
  br label %212

212:                                              ; preds = %207, %211
  %213 = phi float [ 0.000000e+00, %211 ], [ %210, %207 ]
  br label %214

214:                                              ; preds = %212
  %215 = fadd float %199, %213
  %216 = add i64 %198, 1
  br label %197

217:                                              ; preds = %197
  %218 = getelementptr float, ptr %187, i64 %194
  store float %199, ptr %218, align 4
  %219 = add i64 %194, 1
  br label %193

220:                                              ; preds = %193
  %221 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %222 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %221, 0
  %223 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %222, ptr %221, 1
  %224 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %223, i64 0, 2
  %225 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %224, i64 1024, 3, 0
  %226 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %225, i64 1, 4, 0
  br label %227

227:                                              ; preds = %230, %220
  %228 = phi i64 [ %240, %230 ], [ 0, %220 ]
  %229 = icmp slt i64 %228, 1024
  br i1 %229, label %230, label %241

230:                                              ; preds = %227
  %231 = getelementptr float, ptr %187, i64 %228
  %232 = load float, ptr %231, align 4
  %233 = getelementptr float, ptr %137, i64 %228
  %234 = load float, ptr %233, align 4
  %235 = fmul float %234, %234
  %236 = fsub float %232, %235
  %237 = fcmp olt float %236, 0.000000e+00
  %238 = select i1 %237, float 0.000000e+00, float %236
  %239 = getelementptr float, ptr %221, i64 %228
  store float %238, ptr %239, align 4
  %240 = add i64 %228, 1
  br label %227

241:                                              ; preds = %227
  %242 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %243 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %242, 0
  %244 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %243, ptr %242, 1
  %245 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %244, i64 0, 2
  %246 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %245, i64 1024, 3, 0
  %247 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %246, i64 1, 4, 0
  br label %248

248:                                              ; preds = %251, %241
  %249 = phi i64 [ %256, %251 ], [ 0, %241 ]
  %250 = icmp slt i64 %249, 1024
  br i1 %250, label %251, label %257

251:                                              ; preds = %248
  %252 = getelementptr float, ptr %221, i64 %249
  %253 = load float, ptr %252, align 4
  %254 = call float @llvm.sqrt.f32(float %253)
  %255 = getelementptr float, ptr %242, i64 %249
  store float %254, ptr %255, align 4
  %256 = add i64 %249, 1
  br label %248

257:                                              ; preds = %248
  %258 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 1024) to i64))
  %259 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %258, 0
  %260 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %259, ptr %258, 1
  %261 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %260, i64 0, 2
  %262 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %261, i64 1024, 3, 0
  %263 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %262, i64 1, 4, 0
  br label %264

264:                                              ; preds = %267, %257
  %265 = phi i64 [ %279, %267 ], [ 0, %257 ]
  %266 = icmp slt i64 %265, 1024
  br i1 %266, label %267, label %280

267:                                              ; preds = %264
  %268 = getelementptr float, ptr %1, i64 %265
  %269 = load float, ptr %268, align 4
  %270 = getelementptr float, ptr %117, i64 %265
  %271 = load float, ptr %270, align 4
  %272 = getelementptr float, ptr %242, i64 %265
  %273 = load float, ptr %272, align 4
  %274 = fsub float %269, %271
  %275 = fcmp olt float %273, 0x3E45798EE0000000
  %276 = select i1 %275, float 0x3E45798EE0000000, float %273
  %277 = fdiv float %274, %276
  %278 = getelementptr float, ptr %258, i64 %265
  store float %277, ptr %278, align 4
  %279 = add i64 %265, 1
  br label %264

280:                                              ; preds = %264
  br label %281

281:                                              ; preds = %284, %280
  %282 = phi i64 [ %288, %284 ], [ 0, %280 ]
  %283 = icmp slt i64 %282, 1024
  br i1 %283, label %284, label %289

284:                                              ; preds = %281
  %285 = getelementptr float, ptr %258, i64 %282
  %286 = load float, ptr %285, align 4
  %287 = getelementptr float, ptr %11, i64 %282
  store float %286, ptr %287, align 4
  %288 = add i64 %282, 1
  br label %281

289:                                              ; preds = %281
  ret void
}

define void @_mlir_ciface_qfx_kernel(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %0, align 8
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 4, 0
  %10 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 2
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 3, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 4, 0
  call void @qfx_kernel(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
