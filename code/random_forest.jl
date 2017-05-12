using Images
using DataFrames
using DecisionTree
using JLD #using JLD to store model once trained

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#20x20 pixels images, so imageSize is set to 400.
#path set to the location of the data files.

counter = 0 #counter

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo[1]) 
  #Read image file 
  nameFile = "$(path)/$(typeData)/$(idImage).Bmp"
  img = load(nameFile)

  temp = convert(Image{Gray}, img)

 global counter = counter + 1

  println("Reading $(typeData) image ... $(counter).Bmp ") #printing counter
    
  #Transform image matrix to a vector and store 
  #it in data matrix 
  x[index, :] = reshape(temp, 1, imageSize)
 end 
 return x
end

#addprocs(4)
#import DecisionTree 			#importing package on each worker
#@everywhere using DecisionTree 	#loading package

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "/home/rushi/Desktop/old_Desktop/BTech_Project/Project/data"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/code/test_labels/trainLabels.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#save("/home/rushi/Desktop/old_Desktop/BTech_Project/Project/data/code/models/RandomForestxTrain.jld", "xTrain", xTrain)


#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/code/test_labels/testLabels1.csv")

#Read test matrix
xTest = read_data("test1", labelsInfoTest, imageSize, path)

#Get only first character of string (convert from string to character).
#Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], labelsInfoTrain[2])

yTrain = convert(Array{Int64,1},yTrain)

tic()
#Convert from character to integer
yTrain = map(Int,yTrain)

#save("/home/rushi/Desktop/old_Desktop/BTech_Project/Project/data/code/models/RandomForestyTrain.jld", "yTrain", yTrain)

model = build_forest(yTrain, xTrain, 20, 100, 1.0, 10)

#save("/home/rushi/Desktop/old_Desktop/BTech_Project/Project/data/code/models/RandomForestModel.jld", "model", model)

predTest = apply_forest(model, xTest)

#Convert integer predictions to character
labelsInfoTest[2] = map(Char,predTest)
toc()

#Save predictions
writetable("$(path)/code/Output/juliaRFSubmission.csv", labelsInfoTest, separator=',', header=true)

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 100, 5, 1.0);
println("5 fold accuracy: $(mean(accuracy))")	

println("Submission file saved in $(path)/code/Output/juliaRFSubmission.csv")

