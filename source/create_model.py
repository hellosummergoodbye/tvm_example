import nnvm.compiler
import nnvm.symbol as sym
import tvm

def main():
    
    x = sym.Variable("x")
    y = sym.Variable("y")
    
    z = sym.matmul(x,y)
    
    compute_graph = nnvm.graph.create(z)
    
    shape = (2,2)
    
    deploy_graph, lib, params = nnvm.compiler.build(compute_graph, target='llvm', shape={"x": shape,"y": shape}, dtype="float32")

    with open("ir.txt",'w') as f:
        
        f.write(deploy_graph.ir())
    
    lib.export_library('module.so')
    
    with open("deploy.json", "w") as f:
        
        f.write(deploy_graph.json())
    

if __name__=="__main__":
    
    main()
    
    
