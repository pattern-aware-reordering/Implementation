
bertin<-function(x,generations=200){

#####################
# local functions
#####################
crossover<-function(bin1, bin2, rows){
 n <- length(bin1)
 breakpoint <- rows+1
 # Durchtrennung VOR breakpoint
 kind1 <- c(bin1[1:(breakpoint - 1)], bin2[breakpoint:n])
 kind2 <- c(bin2[1:(breakpoint - 1)], bin1[breakpoint:n])
 nachkommen <- cbind(kind1,kind2)
 return(nachkommen)
} # end of crossover

mutation.2opt<-function(perm){
 n <- length(perm)
 b <- sort(sample(1:n, 2))
 mut <- perm
 mut[b[1]:b[2]] <- rev(mut[b[1]:b[2]])
 return(mut)
} # end of mutation.2opt

fitness.table<-function(theta,x,rows,cols,stress){
 x <- x[theta[1:rows],theta[(rows+1):(rows+cols)]]
 return(-sum(stress(x)))
} # end of fitness.table

stress<-function(x){
# x must be matrix!
if(is.matrix(x)==FALSE){
   cat("The argument is not a matrix");return(0)
}
z<-dim(x)[1]; s<-dim(x)[2]

x1<-rbind(x[-1,],x[z,])
x2<-rbind(x[1,],x[-z,])
x3<-cbind(x[,-1],x[,s])
x4<-cbind(x[,1],x[,-s])

x5<-x[-1,-1]
x5<-rbind(x5,x[z,-1])
x5<-cbind(x5,x5[,(s-1)])
x5[z,]<-x[z,];x5[,s]<-x[,s]


x6<-x[-1,-s]
x6<-rbind(x6,x[z,-s])
x6<-cbind(x6[,1],x6)
x6[z,]<-x[z,];x6[,1]<-x[,1]

x7<-x[-z,-1]
x7<-rbind(x[1,-1],x7)
x7<-cbind(x7,x[,s])
x7[1,]<-x[1,];x7[,s]<-x[,s]

x8<-x[-z,-s]
x8<-rbind(x[1,-s],x8)
x8<-cbind(x[,1],x8)
x8[1,]<-x[1,];x8[,1]<-x[,1]

st<-abs(x-x1)+abs(x-x2)+abs(x-x3)+abs(x-x4)+abs(x-x5)+abs(x-x6)+abs(x-x7)+abs(x-x8)
return(st)
} # end of stress

############################
# Initialization,
# Program Start
############################

.pmut<-0.5
.pcross<-0.5
.popsize<-20

rows<-dim(x)[1]
cols<-dim(x)[2]
image(x)

# Start Population

.POP<-c(sample(rows),sample(cols))
for(i in 1:(.popsize-1)) .POP<-cbind(.POP,c(sample(rows),sample(cols)))

.fitness<-apply(.POP,2,fitness.table,x,rows,cols,stress)

##############################
# Iteration Start
##############################

for(i in 1:generations){
 .best<-.POP[,order(.fitness)[.popsize]]

 # MUTATION
 .POP1<-.POP[1:rows,]
 .POP2<-.POP[(rows+1):(rows+cols),]
 .welche<-as.logical(rbinom(.popsize,1,.pmut))
 if(sum(.welche)>1) .POP1[,.welche]<-apply(.POP1[,.welche],2,mutation.2opt)
 .welche<-as.logical(rbinom(.popsize,1,.pmut))
 if(sum(.welche)>1) .POP2[,.welche]<-apply(.POP2[,.welche],2,mutation.2opt)

 .POP<-rbind(.POP1,.POP2)

 # CROSSOVER
 .cr.anzahl<-rbinom(1,.popsize, .pcross)
 .zaehler<-1
 repeat{
   if(.zaehler>.cr.anzahl){
     break
   } else {
     .welche<-sample(1:.popsize,2)
     .POP[,.welche]<-crossover(.POP[,.welche[1]],.POP[,.welche[2]],rows)
     .zaehler<-.zaehler+1
   }
 }
 .fitness<-apply(.POP,2,fitness.table,x,rows,cols,stress)

 # ELITE TOURANMENT SELECTION
 .POP.neu<-.POP
 for(j in 1:.popsize)    {
   .auswahl<-sample(1:.popsize,2)
   .index<-.auswahl[order(apply(.POP[,.auswahl],2,fitness.table,x,rows,cols,stress))[2]]
   .POP.neu[,j]<-.POP[,.index]
 }
 .POP<-.POP.neu

 .POP[,1]<-.best

 .fitness<-apply(.POP,2,fitness.table,x,rows,cols,stress)
 cat(i,":",max(.fitness),"\n")

 .tabelle<-x[.POP[1:rows,1],.POP[(rows+1):(rows+cols)]]
 image(.tabelle)
 title(main=paste("fitness: ",max(.fitness)))
} # end of for loop

##############################
# End Iteration Step
##############################

table.reordered<-x[.POP[1:rows,1],.POP[(rows+1):(rows+cols)]]
print(table.reordered)
invisible(table.reordered)
} # end of function bertin


##############################
# Function "bertin" has to be called with
# the table you want to optimize
##############################


##############################
# Example: Logical matrix   bertin.mat
##############################
bertin.mat<-c(0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,
             ,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,
             ,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,
             ,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,
             ,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,
             ,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,1,
             ,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,
             ,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,
             ,0,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0)
bertin.mat<-matrix(bertin.mat,nrow=9,byrow=TRUE)
dimnames(bertin.mat)<-list(c(
   "High School","Agricultural Coop.","Railway Station",
   "One-Room-School", "Veterinary" ,"No Doctor",
   "No Water Supply" ,"Police Station","Land Reallocation"
),c(LETTERS[1:16]))
