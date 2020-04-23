# This script contains routines to generate the illustrations for the paper "Geometry of v-Tamari lattices in types A and B", written in collaboration with Cesar Ceballos and Arnau Padrol and published in Transactions of the American Mathematical Society, 371 (2019) 2575-2622 (https://doi.org/10.1090/tran/7405)

# Below are a couple of test cases used for debugging purposes

#forests=[[(0,1),(0,10),(2,4),(3,4),(5,7),(6,7),(8,10),(9,10)], [(0,1),(0,10),(2,4),(3,4),(5,10),(6,7),(8,10),(9,10)], [(0,1),(0,10),(2,7),(3,4),(5,7),(6,7),(8,10),(9,10)], [(0,1),(0,10),(2,10),(3,4),(5,10),(6,7),(8,10),(9,10)], [(0,1),(0,10),(2,10),(3,4),(5,7),(6,7),(8,10),(9,10)]]

#forests=[[(0,10),(1,3),(2,3),(4,6),(5,6),(7,9),(8,9)], [(0,10),(1,3),(2,3),(4,6),(5,6),(7,10),(8,9)], [(0,10),(1,3),(2,3),(4,10),(5,6),(7,9),(8,9)], [(0,10),(1,10),(2,3),(4,6),(5,6),(7,9),(8,9)], [(0,10),(1,3),(2,3),(4,9),(5,6),(7,9),(8,9)], [(0,10),(1,9),(2,3),(4,6),(5,6),(7,9),(8,9)], [(0,10),(1,6),(2,3),(4,6),(5,6),(7,9),(8,9)], [(0,10),(1,3),(2,3),(4,10),(5,6),(7,10),(8,9)], [(0,10),(1,10),(2,3),(4,6),(5,6),(7,10),(8,9)], [(0,10),(1,10),(2,3),(4,10),(5,6),(7,9),(8,9)], [(0,10),(1,10),(2,3),(4,10),(5,6),(7,10),(8,9)], [(0,10),(1,9),(2,3),(4,9),(5,6),(7,9),(8,9)], [(0,10),(1,6),(2,3),(4,6),(5,6),(7,10),(8,9)], [(0,10),(1,10),(2,3),(4,9),(5,6),(7,9),(8,9)]]

#I=[0,1,2,4,5,7,8]
#J=[3,6,9,10]


#forests=[[(0,7),(1,2),(3,4),(5,6)]]
#I=[0,1,3,5]
#J=[2,4,6,7]

#cyclicForest=[[(0,0),(1,1),(2,2),(3,3)]]

#cyclicForest=[[(0,2),(1,2),(3,2),(4,5),(6,2),(7,8)], [(0,2),(1,2),(3,5),(4,5),(6,2),(7,8)], [(0,2),(1,2),(3,5),(4,5),(6,5),(7,8)], [(0,5),(1,2),(3,5),(4,5),(6,5),(7,8)], [(0,5),(1,2),(3,5),(4,5),(6,8),(7,8)], [(0,2),(1,2),(3,5),(4,5),(6,8),(7,8)], [(0,2),(1,2),(3,2),(4,5),(6,8),(7,8)], [(0,2),(1,2),(3,8),(4,5),(6,8),(7,8)], [(0,8),(1,2),(3,5),(4,5),(6,8),(7,8)], [(0,8),(1,2),(3,8),(4,5),(6,8),(7,8)]]

#cyclicForest=[[(0, 1), (2, 4), (3, 4), (5, 4), (6, 7)], [(0, 1), (2, 1), (3, 4), (5, 7), (6, 7)], [(0, 1), (2, 1), (3, 4), (5, 1), (6, 7)], [(0, 1), (2, 7), (3, 4), (5, 7), (6, 7)], [(0, 1), (2, 4), (3, 4), (5, 7), (6, 7)], [(0, 1), (2, 4), (3, 4), (5, 1), (6, 7)]]

#cyclicForest=[[(0, 1), (2, 10), (3, 4), (5, 7), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 10), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 1), (3, 4), (5, 7), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 1), (3, 4), (5, 10), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 1), (3, 4), (5, 1), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 7), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 1), (3, 4), (5, 1), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 4), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 1), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 7), (3, 4), (5, 7), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 1), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 7), (3, 4), (5, 7), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 4), (6, 7), (8, 4), (9, 10)], [(0, 1), (2, 1), (3, 4), (5, 7), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 10), (3, 4), (5, 10), (6, 7), (8, 10), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 4), (6, 7), (8, 1), (9, 10)], [(0, 1), (2, 7), (3, 4), (5, 7), (6, 7), (8, 7), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 7), (6, 7), (8, 7), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 7), (6, 7), (8, 4), (9, 10)], [(0, 1), (2, 4), (3, 4), (5, 7), (6, 7), (8, 10), (9, 10)]]


def cross(e0,e1):
    #receives two edges of complete bipartite graph. returns True
    #if they cyclically cross and False otherwise
    n=max(max(e0[0],e0[1]),max(e1[0],e1[1]))

    if ((e0[1]-e1[0])%(n+1) < (e0[1]-e0[0])%(n+1) and (e1[1]-e1[0])%(n+1)>(e0[1]-e1[0])%(n+1)) or ((e1[1]-e0[0])%(n+1)<(e0[1]-e0[0])%(n+1) and (e1[1]-e1[0])%(n+1)>(e1[1]-e0[0])%(n+1)): return True
    else: return False

def coveringForests(I,J):
    #receives as input two subsets I,J, produces first graph of cyclically
    #crossing edges of the complete bipartite graph K_{I,J},  then its
    #complement, and then the clique complex of that complement, which
    #is the simplicial complex whose simplices consist of pairwise cyclically
    #noncrossing edges. then it computes the boundary, and the minimal faces
    # not in the boundary, which are the covering forests

    # returns a list of covering forests

    #first we compute the cyclically noncrossing complex:
    
    n=max(max(I),max(J))
    
    V=[(i,j) for i in I for j in J] # usar este para cicloedros
    #V=[(i,j) for i in I for j in J if i<= j] # usar este para asociahedros

    crossingGraph=Graph([V,cross])

    nonCrossingGraph=crossingGraph.complement()

    ncComplex=nonCrossingGraph.clique_complex()

        
    #second we compute its interior faces:

    
    d=ncComplex.dimension()
    
    codim1=ncComplex.n_skeleton(d-1)
    

    codim1Bdy=[]
    
    for f in codim1.facets():
        Link=ncComplex.link(f)
        if len(Link.facets())==1: codim1Bdy.append(list(f))
    
    Boundary=SimplicialComplex(codim1Bdy)

    nonBdy=[sorted(list(F)) for F in Boundary.minimal_nonfaces()]

    crossings=[sorted(list(E)) for E in crossingGraph.edges(labels=False)]



    CF=[cf for cf in nonBdy if cf not in crossings]

    return CF

    #coveringForests=ncComplex.faces(subcomplex=Boundary)

    #print coveringForests

def getInequalities(F):
    # computes the inequality description, in polymake format (a matrix), of the bounded
    # cell corresponding to covering forest F
    
    I=[]
    J=[]
    for arc in F:
        if arc[0] not in I: I.append(arc[0])
        if arc[1] not in J: J.append(arc[1])
    I.sort()
    J.sort()
    

    # An inequality a0 + a1 x1 + … + ad xd >= 0 is encoded as a row vector (a0,a1,…,ad)
    ineqs=[]
    for a in F:        
        for j in J:
            if j >= a[0] and j != a[1]:
                row=[1 if J.index(a[1])==jp else -1 if J.index(j)==jp else 0 for jp in range(len(J))]
                # hij=-(j-a[0])^2+(a[1]-a[0])^2
                hij=numerical_approx(sqrt((j-a[0]))-sqrt((a[1]-a[0])))
                row=[hij]+row
                # print row
                ineqs.append(row)
    cero=[[0]+[-1 if jp==len(J)-1 else 0 for jp in range(len(J))], [0]+[1 if jp==len(J)-1 else 0 for jp in range(len(J))]]
    ineqs=ineqs+cero
    return ineqs

def nonCrossingHeight(i,j,n):
    # evaluates a cyclically non-crossing height function at a pair i, \ol j
    # where the period of cyclicity is n
    #f= numerical_approx(10*log((j-i)%(n+1)))
    f=(-1)*numerical_approx(3^(-((j-i)%(n+1))))
    return f

def getCyclic(F):
    # this generates the cyclic inequalities corresponding to cyclic covering forest F
    I=[]
    J=[]
    
    
    for arc in F:
        if arc[0] not in I: I.append(arc[0])
        if arc[1] not in J: J.append(arc[1])
    I.sort()
    J.sort()

    n=max(max(I),max(J))
    
    
    # computes the inequality description, in polymake format (a matrix), of the bounded
    # cell corresponding to covering forest F
    # An inequality a0 + a1 x1 + … + ad xd >= 0 is encoded as a row vector (a0,a1,…,ad)
    ineqs=[]
    for a in F:        
        for j in J:
            #if j >= a[0] and j != a[1]: # usar este para asociaedros
            if j != a[1]: # usar este para cicloedros
                row=[1 if J.index(a[1])==jp else -1 if J.index(j)==jp else 0 for jp in range(len(J))]
                hij=nonCrossingHeight(a[0],j,n)-nonCrossingHeight(a[0],a[1],n)
                
                row=[hij]+row
                # print row
                ineqs.append(row)
    cero=[[0]+[-1 if jp==len(J)-1 else 0 for jp in range(len(J))], [0]+[1 if jp==len(J)-1 else 0 for jp in range(len(J))]]
    ineqs=ineqs+cero
    return ineqs

# this still has to be extended like write cyclohedron
def writeAssociahedron(coveringFs):
    # takes a list of covering forests and outputs polymake code with inequality
    # descriptions, definition of polytopes and visualization of polytopes
    m = len(coveringFs)
    
    
    matrices=''
    politopos=''
    imprimirlos='compose('+','.join(['$pproj'+str(k)+'->VISUAL(VertexStyle=>"hidden")' for k in range(m)])+');'

    
    
    for k in range(m):
        ineqs=getInequalities(coveringFs[k])
        

        matrices=matrices+'$a'+str(k)+'= new Matrix<Rational>('+str(ineqs)+');'

        politopos=politopos+'$p'+str(k)+'= new Polytope<Rational>(INEQUALITIES=>$a'+str(k)+');$pproj'+str(k)+'=projection_full($p'+str(k)+');'

    # change this in accordance with case being computed    
    with open('2cyclohedron4.txt','w') as archivo:

        archivo.write(matrices+'\n')
        archivo.write(politopos+'\n')
        archivo.write('\n')
        archivo.write('-------------------------------\n')
        archivo.write(imprimirlos+'\n')
        
        
def writeCyclohedron(I,J):
    # takes a list of covering forests and outputs polymake code with inequality
    # descriptions, definition of polytopes and visualization of polytopes
    coveringFs=coveringForests(I,J)
    m = len(coveringFs)

    minkowskiSums=[]

    for F in range(len(coveringFs)):
        
        oneSum=[]
        
        for i in I:
            
            ithSummand=[]
            
            for e in coveringFs[F]:
                
                if e[0]==i: ithSummand.append(e[1])

            oneSum.append(ithSummand)
            
        minkowskiSums.append(oneSum)

    
        
    matrices=''
    politopos=''
    imprimirlos='compose('+','.join(['$pproj'+str(k)+'->VISUAL(VertexStyle=>"hidden")' for k in range(m)])+');'

    
    
    for k in range(m):
       
        ineqs=getCyclic(coveringFs[k])

        matrices=matrices+'$a'+str(k)+'= new Matrix<Rational>('+str(ineqs)+');'

        politopos=politopos+'$p'+str(k)+'= new Polytope<Rational>(INEQUALITIES=>$a'+str(k)+');$pproj'+str(k)+'=projection_full($p'+str(k)+');'

    # change this in accordance with case being computed    
    with open('cyclohedron-power-'+''.join([str(i) for i in I])+'-'+''.join([str(j) for j in J])+'.txt','w') as archivo:

        archivo.write(matrices+'\n')
        archivo.write(politopos+'\n')
        archivo.write('\n')
        archivo.write('-------------------------------\n')
        archivo.write(imprimirlos+'\n')
        archivo.write('\n')
        archivo.write(',\n'.join([str(w)+': '+str(minkowskiSums[w]) for w in range(len(minkowskiSums))]))

    
def mixedSubdivision(I,J):
    # receives a pair of index sets I,J and produces the representation
    # of the noncrossing complex as a fine mixed subdivision of a
    # generalized permutahedron/polymatroid

    n=len(J)-1

    
    #V=[(i,j) for i in I for j in J] # usar este para cicloedros
    V=[(i,j) for i in I for j in J if i<= j] # usar este para asociahedros

    # first we compute the clique complex of the non-crossing graph 

    crossingGraph=Graph([V,cross])

    nonCrossingGraph=crossingGraph.complement()

    ncComplex=nonCrossingGraph.clique_complex()

    # now we get minkowski sum representations for the facets of
    # the noncrossing complex

    minkowskiSums=[]

    for F in ncComplex.facets():
        
        oneSum=[]
        
        for i in I:
            
            ithSummand=[]
            
            for e in F:
                
                if e[0]==i: ithSummand.append(e[1])

            oneSum.append(ithSummand)
            
        minkowskiSums.append(oneSum)


    # now generate polymake code with the minkowski sums giving
    # the fine mixed subdivision

    stdSimplexVertices=[v.vector() for v in polytopes.simplex(n,project=True).vertices()]

    mixedCells=[sum([Polyhedron(vertices=[stdSimplexVertices[J.index(v)] for v in sumando]) for sumando in c]) for c in minkowskiSums]

    mixedCellsPoly=';'.join(['; '.join(['$s'+str(i)+'=new Polytope<Rational>(POINTS=>'+str([[1]+list(stdSimplexVertices[J.index(v)]) for v in minkowskiSums[c][i]])+')' for i in range(len(minkowskiSums[c]))])+';'+'$p'+str(c)+'=minkowski_sum_fukuda('+','.join(['$s'+str(j) for j in range(len(minkowskiSums[c]))])+')'  for c in range(len(minkowskiSums))])+';'

    
    #smallSimplex=[v.vector() for v in polytopes.simplex(n-1,project=True).vertices()]
    #restrictedMinkowskiSums=[[[i for i in e if i!= 0] for e in S] for S in minkowskiSums]
    
    #restrictedMixedCellsPoly=';'.join(['; '.join(['$s'+str(i)+'=new Polytope<Rational>(POINTS=>'+str([[1]+list(smallSimplex[v-1]) for v in restrictedMinkowskiSums[c][i]])+')' for i in range(len(restrictedMinkowskiSums[c]))])+';'+'$p'+str(c)+'=minkowski_sum_fukuda('+','.join(['$s'+str(j) for j in range(len(restrictedMinkowskiSums[c]))])+')'  for c in range(len(restrictedMinkowskiSums))])+';'



    toPolyMatrix=';'.join(['$a'+str(c)+'=new Matrix<Rational>(['+','.join([str([1]+list(+v.vector())) for v in mixedCells[c].vertices()])+'])' for c in range(len(mixedCells))])+';'

    toPolyPolytope=';'.join(['$p'+str(c)+'=new Polytope<Rational>(POINTS=>$a'+str(c)+')' for c in range(len(mixedCells))])+';'

    printPoly='compose('+','.join(['$p'+str(c)+'->VISUAL(VertexStyle=>"hidden")' for c in range(len(mixedCells))])+');'

    # print the output to a file

    with open('mixedSubdivision'+''.join([str(i)for i in I])+'-'+''.join([str(j) for j in J])+'.txt','w') as archivo:

        archivo.write(mixedCellsPoly+'\n')
        archivo.write('\n')
        archivo.write('-------------------------------\n')
        archivo.write(printPoly+'\n')
        archivo.write('\n')
        archivo.write(',\n'.join([str(w)+': '+str(minkowskiSums[w]) for w in range(len(minkowskiSums))]))

        

def nonKCrossing(I,J,k):
    # receives a pair of index sets I,J and produces the representation
    # of the non-k-crossing complex as if it were a fine mixed subdivision of a
    # generalized permutahedron/polymatroid.
    # default for triangulations is k=1

    n=len(J)-1

    
    #V=[(i,j) for i in I for j in J] # usar este para cicloedros
    V=[(i,j) for i in I for j in J if i<= j] # usar este para asociahedros

    # first we compute the clique complex of the crossing graph 

    crossingGraph=Graph([V,cross])

    crossingComplex=crossingGraph.clique_complex()

    #initialize the complex of non-k-crossing edges as a full simplex
    #on the edges of the graph
    
    nonKCrossingComplex=SimplicialComplex([V])


    
    for nonface in crossingComplex.n_cells(k):
        nonKCrossingComplex.remove_face(nonface)

    G=nonKCrossingComplex.flip_graph()
        
############## uncomment to generate fake mixed subdivision ####
    # now we get fake minkowski sum representations for the facets of
    # the noncrossing complex

    minkowskiSums=[]

    for F in nonKCrossingComplex.facets():
        
        oneSum=[]
        
        for i in I:
            
            ithSummand=[]
            
            for e in F:
                
                if e[0]==i: ithSummand.append(e[1])

            oneSum.append(ithSummand)
            
        minkowskiSums.append(oneSum)

############## uncomment to generate fake mixed subdivision ####   
    # now generate polymake code with the minkowski sums giving
    # the fine mixed subdivision

    stdSimplexVertices=[v.vector() for v in polytopes.simplex(n,project=True).vertices()]

    mixedCells=[sum([Polyhedron(vertices=[stdSimplexVertices[J.index(v)] for v in sumando]) for sumando in c]) for c in minkowskiSums]

    mixedCellsPoly=';'.join(['; '.join(['$s'+str(i)+'=new Polytope<Rational>(POINTS=>'+str([[1]+list(stdSimplexVertices[J.index(v)]) for v in minkowskiSums[c][i]])+')' for i in range(len(minkowskiSums[c]))])+';'+'$p'+str(c)+'=minkowski_sum_fukuda('+','.join(['$s'+str(j) for j in range(len(minkowskiSums[c]))])+')'  for c in range(len(minkowskiSums))])+';'

    toPolyMatrix=';'.join(['$a'+str(c)+'=new Matrix<Rational>(['+','.join([str([1]+list(+v.vector())) for v in mixedCells[c].vertices()])+'])' for c in range(len(mixedCells))])+';'

    toPolyPolytope=';'.join(['$p'+str(c)+'=new Polytope<Rational>(POINTS=>$a'+str(c)+')' for c in range(len(mixedCells))])+';'

    printPoly='compose('+','.join(['$p'+str(c)+'->VISUAL(VertexStyle=>"hidden")' for c in range(len(mixedCells))])+');'

    # print the output to a file

    with open('non'+str(k)+'Crossing'+''.join([str(i)for i in I])+'-'+''.join([str(j) for j in J])+'.txt','w') as archivo:

        archivo.write(mixedCellsPoly+'\n')
        archivo.write('\n')
        archivo.write('-------------------------------\n')
        archivo.write(printPoly+'\n')
        archivo.write('\n')
        archivo.write(',\n'.join([str(w)+': '+str(minkowskiSums[w]) for w in range(len(minkowskiSums))]))

##########################

    with open('dualgraphk'+str(k)+'-I'+''.join([str(i) for i in I])+'-J'+''.join([str(j) for j in J])+'.txt','w') as grafo:
        
        for i in range(len(G.vertices())):
            grafo.write(str(i)+': '+str(G.vertices()[i])+'\n')

        G.relabel(perm=None)
        grafoPolymake=[list(e) for e in G.edges(labels=False)]

        grafo.write('$g=graph_from_edges('+str(grafoPolymake)+');\n')
        grafo.write('$g->VISUAL;\n')

    return nonKCrossingComplex
