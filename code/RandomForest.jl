Pkg.add("Images")
Pkg.add("DataFrames")
Pkg.add("DecisionTree")

using Images
using DataFrames
using DecisionTree

function read_data(typeData, labelsInfo, imageSize, path)
 	#Intialize x matrix
	
	x = zeros(size(labelsInfo, 1), imageSize)
	
	for (index, idImage) in enumerate(labelsInfo[1]) 
		#Read image file 
		nameFile = "$(path)/$(typeData)/$(idImage).Bmp"
		img = load(nameFile)
		temp = convert(Image{Gray}, img)
		x[index, :] = reshape(temp, 1, imageSize)
	end 
	
	return x
end

imageSize = 400 # 20 x 20 pixel

path = "/home/rushi/Desktop/BTech_Project/Project/data"

labelsInfoTrain = readtable("$(path)/trainLabels.csv")

xTrain = read_data("train", labelsInfoTrain, imageSize, path)

labelsInfoTest = readtable("$(path)/testLabels.csv")

xTest = read_data("test", labelsInfoTest, imageSize, path)

yTrain = map(x -> x[1], labelsInfoTrain[2])

yTrain = convert(Array{Int64,1},yTrain)

yTrain = map(Int,yTrain)

#building Radom_forest
model = build_forest(yTrain, xTrain, 20, 50, 1.0)

predTest = apply_forest(model, xTest)

#Convert integer predicted to character
labelsInfoTest[2] = map(Char,predTest)

writetable("juliaSubmissionRF.csv", labelsInfoTest, separator=',', header=true)

accuracy = nfoldCV_forest(yTrain, xTrain, 20, 50, 3, 1.0);
println("accuracy: $(mean(accuracy))")	

