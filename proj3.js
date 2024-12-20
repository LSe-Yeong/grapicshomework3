import*as THREE from 'three'
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {vec2, vec3, vec4, mat4, utils} from 'wgpu-matrix';

const loc_position = 3;
const loc_normal = 2;
const loc_inter_stage_normal = 1;

const id_group = 0;
const binding_matrices = 7;

const format_depth_texture = 'depth24plus';

const tankState = {
    position: vec3.create(0,0,0),  // 초기 위치 (x, y, z)
    rotation: 0,  // 초기 회전 (라디안 단위)
};
const turretState = {
    rotation : vec3.create(0,0,0)
}
const gunState = {
    rotation : vec3.create(0,0,0)
}

async function main() {

    //초기 설정
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice();
    if(!device) {
        throw Error("WebGPU not supported.");
    }

    const canvas = document.querySelector("#webgpu");
    const context = canvas.getContext("webgpu");
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: preferredFormat,
    });

    let canvasTexture = context.getCurrentTexture();

    // UI 설정 (EventHandler 역할)
    class UI {
        static NONE = 0; //none
        static ROTATING = 1; //회전
        static TRANSLATING = 2; //이동
        static mouseMove = UI.NONE; //현재 마우스 상태
        static camera = {fovy:60, position:vec3.create(0,0,-3), near:0.1, far:100}; //시야각, 위치, 클리핑 평면
        static matrices = { //P(원근 투영 행렬),R(회전) 행렬
            P: mat4.perspective(utils.degToRad(UI.camera.fovy), canvasTexture.width/canvasTexture.height, UI.camera.near, UI.camera.far),
            R: mat4.identity(),
            M: mat4.identity(),  // 탱크의 모델 행렬
        };
        static onmousedown(ev) {
            if(ev.buttons & 1)  { UI.mouseMove = UI.ROTATING; } //마우스 눌렀을 때 회전 모드(왼쪽)
            else if(ev.buttons & 4) { UI.mouseMove = UI.TRANSLATING ; } //마우스 눌렀을 때 이동 모드(오른쪽)
        };
        static onmouseup(ev) {
            UI.mouseMove = UI.NONE; //떼면 아무 상태도 아닌거로 돌아옴
        };
        static onmousemove(ev) { //마우스를 움직일 때
            let offset = [ev.movementX, ev.movementY];
            if (UI.mouseMove == UI.ROTATING) {
                UI.update_VP(); //VP 업데이트 하고
                let axis = unproject_vector([offset[1], offset[0], 0], UI.matrices.VP,
                    [0, 0, canvas.clientWidth, canvas.clientHeight]); //마우스 이동 방향에 따른 회전 축 계산
                UI.matrices.R = mat4.rotate(UI.matrices.R, [axis[0], axis[1], axis[2]], utils.degToRad(vec2.lenSq(offset)*0.1));  //회전 행렬 업데이트
            }
            else if(UI.mouseMove == UI.TRANSLATING) {
                UI.update_VP();
                let by = unproject_vector([offset[0], -offset[1], 0], UI.matrices.VP,
                    [0, 0, canvas.clientWidth, canvas.clientHeight]); //새로운 위치벡터
                UI.camera.position = vec3.add(UI.camera.position, vec3.transformMat4(vec3.create(by[0], by[1], by[2]), UI.matrices.R));
                //카메라 위치에 위치 벡터 더함
            }
        };
        static onwheel(ev) {
            ev.preventDefault();
            UI.camera.position[2] = -Math.max(1, Math.min(-UI.camera.position[2] + ev.deltaY*0.01, 50)); //카메라의 Z값 위치 변경
            UI.update_VP();
        };
        static onkeydown(ev) {
            const moveStep = 0.1; // 이동 거리
            const rotateStep = Math.PI / 36; // 회전 각도 (5도)
        
            switch (ev.key) {
                case "ArrowUp": // 전진
                    // 탱크가 바라보는 방향으로 전진
                    tankState.position[0] += moveStep * Math.cos(tankState.rotation);
                    tankState.position[2] += moveStep * Math.sin(-tankState.rotation);
                    break;
                case "ArrowDown": // 후진
                    // 탱크가 바라보는 방향 반대쪽으로 후진
                    tankState.position[0] -= moveStep * Math.cos(tankState.rotation);
                    tankState.position[2] -= moveStep * Math.sin(-tankState.rotation);
                    break;
                case "ArrowLeft": // 좌회전
                    // 회전
                    tankState.rotation += rotateStep;
                    // turretState.rotation += rotateStep;
                    // gunState.rotation += rotateStep;
                    break;
                case "ArrowRight": // 우회전
                    // 회전
                    tankState.rotation -= rotateStep;
                    // turretState.rotation -= rotateStep;
                    // gunState.rotation -= rotateStep;
                    console.log(turretState)
                    console.log(tankState)
                    break;
                case "a":
                    turretState.rotation[1] += rotateStep
                    gunState.rotation[1] += rotateStep
                    break;
                case "d":
                    turretState.rotation[1] -= rotateStep
                    gunState.rotation[1] -= rotateStep
                    break;
                case "w":
                    gunState.rotation[2] += rotateStep
                    break;
                case "s":
                    gunState.rotation[2] -= rotateStep
                    break;

            }
            console.log(tankState)
            console.log(turretState)
            console.log(gunState)
            updateModelMatrix();
        }

        static update_VP() {
            UI.matrices.VP = mat4.multiply(mat4.translate(UI.matrices.P, UI.camera.position),UI.matrices.R); //VP 행렬 업데이트
        }
    };

    UI.update_VP();

    canvas.onmousedown = UI.onmousedown;
    canvas.onmouseup = UI.onmouseup;
    canvas.onmousemove = UI.onmousemove;
    window.addEventListener("wheel", UI.onwheel, {passive:false});
    window.addEventListener("keydown", UI.onkeydown, {passive:false});

    function updateModelMatrix() {
        // 단위 행렬 생성
        let modelMatrix = mat4.identity(); // 단위 행렬
    
        console.log("Initial Identity Matrix:", modelMatrix);  // 단위 행렬 확인

        const tankStateRotation = tankState.rotation;
    
        // 위치 변환
        modelMatrix = mat4.translate(modelMatrix, tankState.position);  // 위치 변환 (modelMatrix에 직접 적용)
        console.log("After Translation:", modelMatrix);  // 위치 변환 후 행렬 확인
    
        // Y축 회전
        modelMatrix = mat4.rotateY(modelMatrix, tankStateRotation);  // 회전 (modelMatrix에 직접 적용)
        console.log("After Rotation:", modelMatrix);  // 회전 후 행렬 확인
    
        // 최종 모델 행렬을 UI.matrices.M에 저장
        UI.matrices.M = modelMatrix;
    }

    //탱크 로드
    const tank = await load_gltf("/resource/tank.glb",device, preferredFormat,0);
    console.log(tank)

    //포탑, 건 로드
    const turret = await load_gltf("/resource/tank.glb",device, preferredFormat,2)
    
    const gun = await load_gltf("/resource/tank.glb",device, preferredFormat,1)

    //grid 로드
    const grid = load_grid(device,preferredFormat)

    //world axis 로드
    const worldAxis = load_worldAxis(device,preferredFormat,10,0.5)
    const tankWorldAxis = load_worldAxis(device,preferredFormat,2,1.0)

    console.log(worldAxis)

    //텍스처 생성
    const depthTexture = device.createTexture({ //텍스처의 깊이 설정
        size: [canvasTexture.width, canvasTexture.height],
        format: format_depth_texture,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    let MVP;

    function render(time) {
        
        canvasTexture = context.getCurrentTexture(); //그려질 이미지 cansvas
            
        const encoder = device.createCommandEncoder();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{  //초기 색깔 설정
                view: canvasTexture.createView(),
                loadOp: "clear",
                clearValue: {r:.3, g:.3, b:.3, a:1},
                storeOp: "store",
            }],
            depthStencilAttachment: { //초기 깊이 설정
                view: depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        MVP = mat4.multiply(UI.matrices.VP, UI.matrices.M); // M : 모델 행렬 , V : View(카메라), P : Projection(3D -> 2D)
        // tank.render(renderPass, MVP); //MVP 정보 탱크 렌더링
        // turret.render(renderPass,MVP);
        // gun.render(renderPass,MVP);

        const tankParts=[tank,turret,gun]
        tankPartRender(tankParts,renderPass,MVP)

        grid.render(renderPass, UI.matrices.VP)
        worldAxis.render(renderPass,UI.matrices.VP)
        tankWorldAxis.render(renderPass,MVP)
        renderPass.end();
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);

        requestAnimationFrame(render);
    }

    requestAnimationFrame(render);

}

function tankPartRender(tankParts,renderPass,MVP){
    const tank = tankParts[0]
    const turret = tankParts[1]
    const gun = tankParts[2]

    const center = [tankState.position[0],tankState.position[1],tankState.position[2]]

    tank.render(renderPass, MVP); //MVP 정보 탱크 렌더링
    let newMVP1 = mat4.clone(MVP)
    newMVP1 = mat4.translate(newMVP1,[-0.5,0,0])
    newMVP1 = mat4.rotateY(newMVP1, turretState.rotation[1]);
    newMVP1 = mat4.translate(newMVP1, [0.5,0,0]);
    turret.render(renderPass,newMVP1);

    let newMVP2 = mat4.clone(newMVP1)
    newMVP2 = mat4.translate(newMVP2,[-0.18,0.25,0])
    newMVP2 = mat4.rotateZ(newMVP2,gunState.rotation[2])
    newMVP2 = mat4.translate(newMVP2,[0.18,-0.25,0])
    gun.render(renderPass,newMVP2);

    return 
}

function load_grid(device, preferredFormat) {
    const gridSize = 10; // 그리드 크기 (±gridSize)
    const gridSpacing = 1; // 그리드 간격
    const gridVertices = [];

    // 그리드의 X 축 방향 선 생성
    for (let z = -gridSize; z <= gridSize; z++) {
        gridVertices.push(-gridSize, -0.5, z * gridSpacing); // 시작점
        gridVertices.push(gridSize, -0.5, z * gridSpacing);  // 끝점
    }

    // 그리드의 Z 축 방향 선 생성
    for (let x = -gridSize; x <= gridSize; x++) {
        gridVertices.push(x * gridSpacing, -0.5, -gridSize); // 시작점
        gridVertices.push(x * gridSpacing, -0.5, gridSize);  // 끝점
    }

    const gridBuffer = device.createBuffer({
        label: "grid vertices",
        size: gridVertices.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });

    new Float32Array(gridBuffer.getMappedRange()).set(gridVertices);
    gridBuffer.unmap();

    const module = device.createShaderModule({
        code: `
            struct Uniforms {
                MVP: mat4x4<f32>,
            };

            @binding(0) @group(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn vs_main(@location(0) position: vec3f) -> @builtin(position) vec4f {
                // MVP 행렬로 정점을 변환
                return uniforms.MVP * vec4f(position, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4f {
                return vec4f(0.6, 0.6, 0.6, 1.0); // Grid 색상
            }`
            ,
    });

    const uniformBuffer = device.createBuffer({
        size: 4*4*4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipeline = device.createRenderPipeline({
        label: "grid pipeline",
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs_main",
            buffers: [
                {
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
                    attributes: [
                        {
                            shaderLocation: 0,
                            offset: 0,
                            format: "float32x3",
                        },
                    ],
                },
            ],
        },
        fragment: {
            module,
            entryPoint: "fs_main",
            targets: [{ format: preferredFormat }],
        },
        primitive: {
            topology: "line-list", // 각 두 점이 하나의 선분으로 처리됨
        },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(id_group),
        entries:[
            { binding: 0, resource: {buffer: uniformBuffer} },
        ],
    });

    function render(renderPass,MVP) {
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0,bindGroup);
        renderPass.setVertexBuffer(0, gridBuffer);
        device.queue.writeBuffer(uniformBuffer, 0, MVP);
        renderPass.draw(gridVertices.length / 3); // 총 정점 개수의 1/3만큼 그리기
    }

    return { render };
}

function load_worldAxis(device, preferredFormat,length,colorValue) {
    // 월드 축의 길이를 설정
    const axisLength = length;
    const adjustY = -0.4;
    const axisVertices = [
        // X축 (빨간색)
        -axisLength, adjustY, 0, axisLength, adjustY, 0,
        // Y축 (초록색)
        0, -axisLength+adjustY, 0, 0, axisLength+adjustY, 0,
        // Z축 (파란색)
        0, adjustY, -axisLength, 0, adjustY, axisLength
    ];

    const axisColors = [
        // X축 (빨간색)
        colorValue, 0.0, 0.0,
        colorValue, 0.0, 0.0,
        // Y축 (초록색)
        0.0, colorValue, 0.0,
        0.0, colorValue, 0.0,
        // Z축 (파란색)
        0.0, 0.0, colorValue,
        0.0, 0.0, colorValue
    ];

    const axisBuffer = device.createBuffer({
        label: "axis vertices",
        size: axisVertices.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });

    new Float32Array(axisBuffer.getMappedRange()).set(axisVertices);
    axisBuffer.unmap();

    const colorBuffer = device.createBuffer({
        label: "axis colors",
        size: axisColors.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    
    new Float32Array(colorBuffer.getMappedRange()).set(axisColors);
    colorBuffer.unmap();

    const module = device.createShaderModule({
        code: `
            @group(0) @binding(0) var<uniform> MVP: mat4x4<f32>;

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(2) color: vec3f,
            };

            @vertex
            fn vs_main(@location(0) position: vec3f, @location(1) color: vec3f) -> VertexOutput {
                var output: VertexOutput;
                // MVP 행렬을 사용해 정점을 변환
                output.position = MVP * vec4f(position, 1.0);
                output.color = color; // 색상 값을 전달
                return output;
            }

            @fragment
            fn fs_main(@location(2) color: vec3f) -> @location(0) vec4f {
                // 색상 값을 받아서 반환
                return vec4f(color, 1.0);
            }
        `,
    });


    const uniformBuffer = device.createBuffer({
        size: 4*4*4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const pipeline = device.createRenderPipeline({
        label: "axis pipeline",
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs_main",
            buffers: [
                {
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // 각 정점의 크기 (vec3)
                    attributes: [
                        {
                            shaderLocation: 0, // 셰이더에서 @location(0)으로 참조
                            offset: 0, // 시작 위치
                            format: "float32x3", // vec3f 포맷
                        },
                    ],
                },
                {
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // 색상 크기 (vec3)
                    attributes: [
                        {
                            shaderLocation: 1, // 색상 정보
                            offset: 0,
                            format: "float32x3",
                        },
                    ],
                },
            ],
        },
        fragment: { 
            module, 
            entryPoint: "fs_main", 
            targets: [{ format: preferredFormat }] 
        },
        primitive: { topology: "line-list" },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(id_group),
        entries:[
            { binding: 0, resource: {buffer: uniformBuffer} },
        ],
    });

    function render(renderPass, MVP) {
        // 렌더링
        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.setVertexBuffer(0, axisBuffer);
        renderPass.setVertexBuffer(1, colorBuffer); // 색상 버퍼
        device.queue.writeBuffer(uniformBuffer, 0, MVP);

        renderPass.draw(6, 1, 0, 0); // 정점 6개
    }

    return { render };
}

async function load_gltf(url, device, preferredFormat,meshIdx) {
    const loader = new GLTFLoader();

    const root = await new Promise((resolve,reject) => {
        loader.load(url,
            (model) => {resolve(model);},
            null,
            (error) => {reject(error);});
    });

        // Mesh 객체 탐색
        const meshes = [];
        root.scene.traverse((child) => {
            if (child.isMesh) {
                meshes.push(child);
            }
        });
        
        if (meshes.length === 0) {
            throw new Error("No Mesh objects found in the GLTF model.");
        }
        
        console.log("Mesh objects found:", meshes);
        
        // const vertexBuffers = [];
        // const indexBuffers = [];
        const uniformBuffer = device.createBuffer({
            size: 4 * 4 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        
        // 각 메쉬에 대해 버퍼 생성
        const obj = meshes[meshIdx]        
        if (!obj.geometry || !obj.geometry.attributes) {
            throw new Error("Mesh does not contain geometry attributes.");
        }
    
        const positions = obj.geometry.attributes.position.array;
        const normals = obj.geometry.attributes.normal.array;
        const indices = new Uint32Array(obj.geometry.index.array);
    
        const vertexBuffer = {
            position: device.createBuffer({
                label: `obj${meshIdx} mesh positions`,
                size: positions.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            }),
            normal: device.createBuffer({
                label: `obj${meshIdx} mesh normals`,
                size: normals.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            }),
        };
    
        device.queue.writeBuffer(vertexBuffer.position, 0, positions);
        device.queue.writeBuffer(vertexBuffer.normal, 0, normals);
    
        const indexBuffer = device.createBuffer({
            label: `obj${meshIdx} mesh indices`,
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
    
        device.queue.writeBuffer(indexBuffer, 0, indices);
    
        // vertexBuffers.push(vertexBuffer);
        // indexBuffers.push(indexBuffer);
        
        const shaderModule = device.createShaderModule({
            label: "solid triangle shader",
            code: `
                struct VertexOut {
                    @builtin(position) position: vec4f,
                    @location(0) normal: vec3f
                };
        
                struct Matrices {
                    MVP: mat4x4f,
                };
        
                @group(0) @binding(0) var<uniform> matrices: Matrices;
        
                @vertex fn main_vert(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOut {
                    var vertex: VertexOut;
                    vertex.position = matrices.MVP * vec4f(position, 1.0);
                    vertex.normal = normal;
                    return vertex;
                }
        
                @fragment fn main_frag(@location(0) normal: vec3f) -> @location(0) vec4f {
                    return vec4f(0.5 * (normal + vec3f(1.0)), 1.0);
                }
            `,
        });
        
        const pipeline = device.createRenderPipeline({
            label: "solid triangle pipeline",
            layout: "auto",
            vertex: {
                module: shaderModule,
                entryPoint: "main_vert",
                buffers: [
                    {
                        arrayStride: 4 * 3,
                        attributes: [{ format: "float32x3", offset: 0, shaderLocation: 0 }],
                    },
                    {
                        arrayStride: 4 * 3,
                        attributes: [{ format: "float32x3", offset: 0, shaderLocation: 1 }],
                    },
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: "main_frag",
                targets: [{ format: preferredFormat }],
            },
            primitive: { topology: 'triangle-list' },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: format_depth_texture,
            },
        });
        
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
            ],
        });
        
        function render(renderPass, MVP) {
            // 각 메쉬에 대해 렌더링을 반복

                renderPass.setPipeline(pipeline);
                device.queue.writeBuffer(uniformBuffer, 0, MVP);
                renderPass.setVertexBuffer(0, vertexBuffer.position);
                renderPass.setVertexBuffer(1, vertexBuffer.normal);
                renderPass.setIndexBuffer(indexBuffer, 'uint32');
                renderPass.setBindGroup(0, bindGroup);
                renderPass.drawIndexed(meshes[meshIdx].geometry.index.count);
        };
        
        return { render };
    }        

// https://github.com/g-truc/glm/blob/master/glm/ext/matrix_projection.inl
function project(p_obj, MVP, viewport)
{
    let tmp = vec4.transformMat4(p_obj, MVP);
    tmp = tmp.map((x) => x/tmp[3]); // tmp /= tmp[3]
    for(let i=0 ; i<2 ; i++) {
        tmp[i] = (0.5*tmp[i] + 0.5) * viewport[i+2] + viewport[i];
    }
    return tmp;
}

// https://github.com/g-truc/glm/blob/master/glm/ext/matrix_projection.inl
function unproject(p_win, MVP, viewport) {
    let MVP_inv = mat4.invert(MVP);
    let tmp = mat4.clone(p_win);

    for (let i = 0; i < 2; i++)
        tmp[i] = 2.0 * (tmp[i] - viewport[i]) / viewport[i+2] - 1.0;

    let p_obj = vec4.transformMat4(tmp, MVP_inv);

    p_obj = p_obj.map((x) => x/p_obj[3]);

    return p_obj;
}

function unproject_vector(vec_win, MVP, viewport)
{
    let org_win = project([0,0,0,1], MVP, viewport);
    let vec = unproject([org_win[0]+vec_win[0], org_win[1]+vec_win[1], org_win[2]+vec_win[2], 1],
                        MVP, viewport);
    return vec;
}




main();