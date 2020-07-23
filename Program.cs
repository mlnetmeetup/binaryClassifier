using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using Microsoft.ML.Calibrators;


namespace DemoBinaryClassifier
{
    class Program
    {
        static void Main(string[] args)
        {
            /// Обработаем входные данные

            string busDirectory = Path.Combine(@"..\..\..\..\","DATA", "bus");
            string nobusDirectory = Path.Combine(@"..\..\..\..\","DATA", "no_bus");
            string tf = Path.Combine(@"..\..\..\..\","tf", "tf.pb");


            var bus = Directory.GetFiles(busDirectory).Select(x => new InputModel() { ImagePath = x, IsBus = true }).OrderBy(x => Guid.NewGuid()).Take(100);
            var nobus = Directory.GetFiles(nobusDirectory).Select(x => new InputModel() { ImagePath = x, IsBus = false }).OrderBy(x => Guid.NewGuid()).Take(100);

            MLContext context = new MLContext();

            IDataView data = context.Data.LoadFromEnumerable<InputModel>(bus.Concat(nobus));

            var usedData = context.Data.TrainTestSplit(data,0.3);


            var pipeline = context.Transforms.LoadImages("images", "", "ImagePath")
                .Append(context.Transforms.ResizeImages("resized", 222, 222, "images"))
                .Append(context.Transforms.ExtractPixels("pixels", "resized", interleavePixelColors: true))
                
                .Append(context.Transforms.Conversion.ConvertType("input","pixels"))
                .Append(context.Model.LoadTensorFlowModel(tf)
                                .ScoreTensorFlowModel("softmax2_pre_activation", "input", true)
                )

                .Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression(featureColumnName: "softmax2_pre_activation"));

            Console.WriteLine("start Fitting");
            var model = pipeline.Fit(usedData.TrainSet);
            Console.WriteLine("end Fitting");
            


            var metrics = context.BinaryClassification.Evaluate(model.Transform(usedData.TestSet));

            Console.WriteLine($"Accuracy = {metrics.Accuracy}");

            //Test(context, model, usedData.TestSet);
            context.Model.Save(model, null, "model.zip");
            Console.ReadLine();
            context.Model.Load("model.zip", out var schema);
        }

        private static void Test(MLContext context, TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> model, IDataView testSet)
        {
            var predictions = context.Data.CreateEnumerable<OutputModel>(model.Transform(testSet),true,true);
            foreach (var result in predictions)
                Console.WriteLine($"{result.ImagePath}: real - {result.IsBus}, predicted - {result.PredictedLabel}");

        }
    }
}
