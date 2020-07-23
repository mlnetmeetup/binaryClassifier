using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DemoBinaryClassifier
{
    public class InputModel
    {
        public string ImagePath { get; set; }

        [ColumnName("Label")]
        public bool IsBus { get; set; }
    }

    public class OutputModel : InputModel
    {
        public bool PredictedLabel { get; set; }
    }
}
