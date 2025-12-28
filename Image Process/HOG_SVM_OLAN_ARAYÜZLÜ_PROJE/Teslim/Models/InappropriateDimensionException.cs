using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Teslim.Models
{
    class InappropriateDimensionException : Exception
    {
        public InappropriateDimensionException(string message) : base(message) { }
    }
}
